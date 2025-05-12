import numpy as np
import torch.nn as nn

from . import fastba
from . import altcorr
from .lietorch import SE3

from .extractor import BasicEncoder4
from .blocks import GradientClip, GatedResidual, SoftAgg

from .utils import *
from .ba import BA
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast

DIM = 384

class Update(nn.Module):
    def __init__(self, p):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))
        
        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """

        net = net + inp + self.corr(corr)
        net = self.norm(net)

        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        net = net + self.agg_kk(net, kk)
        net = net + self.agg_ij(net, ii*12345 + jj)

        net = self.gru(net)

        return net, (self.d(net), self.w(net), None)


class Patchifier(nn.Module):
    def __init__(self, patch_size=3):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance')
        self.inet = BasicEncoder4(output_dim=DIM, norm_fn='none')

    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def forward(self, images, patches_per_image=80, disps=None, centroid_sel_strat='RANDOM', return_color=False):
        """
        Gelişmiş yama çıkarma stratejileri ile giriş görüntülerinden yamalar çıkarır.

        Args:
            images: Giriş görüntü tensörü.
            patches_per_image: Her görüntüden çıkarılacak yama sayısı.
            disps: Opsiyonel derinlik/disparite haritası.
            centroid_sel_strat: Yama merkezi seçimi stratejisi. Mevcut seçenekler:
                - 'RANDOM': Uniform rastgele seçim.
                - 'GRADIENT_BIAS': Görüntü gradyanı büyüklüğüne göre seçim.
                - 'EDGE_BIAS': Kenar yoğunluğuna göre seçim (Sobel operatörü kullanılarak hesaplanır).
                - 'GRID': Düzenli ızgara tabanlı seçim.
                - 'GREEDY': Açgözlü algoritma ile, yüksek gradyan değerine sahip ve çakışma olmayan yamalar seçilir.
            return_color: True ise renk bilgisi de döndürülür.

        Returns:
            return_color=True ise (fmap, gmap, imap, patches, index, clr),
            aksi halde (fmap, gmap, imap, patches, index)
        """
        fmap = self.fnet(images) / 4.0
        imap = self.inet(images) / 4.0

        b, n, c, h, w = fmap.shape
        P = self.patch_size
        device = images.device

        if centroid_sel_strat == 'GRADIENT_BIAS':
            g = self.__image_gradient(images)
            x = torch.randint(1, w - 1, size=[n, 3 * patches_per_image], device=device)
            y = torch.randint(1, h - 1, size=[n, 3 * patches_per_image], device=device)
            coords = torch.stack([x, y], dim=-1).float()
            g_vals = altcorr.patchify(g[0, :, None], coords, 0).view(n, 3 * patches_per_image)
            ix = torch.argsort(g_vals, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        elif centroid_sel_strat == 'EDGE_BIAS':
            edge_map = self.__compute_edge_map(images)
            x = torch.randint(1, w - 1, size=[n, 3 * patches_per_image], device=device)
            y = torch.randint(1, h - 1, size=[n, 3 * patches_per_image], device=device)
            coords = torch.stack([x, y], dim=-1).float()
            edge_vals = altcorr.patchify(edge_map[0, :, None], coords, 0).view(n, 3 * patches_per_image)
            ix = torch.argsort(edge_vals, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        elif centroid_sel_strat == 'GRID':
            grid_dim = int(torch.sqrt(torch.tensor(patches_per_image, dtype=torch.float32)).item())
            xs = torch.linspace(1, w - 2, grid_dim, device=device).long()
            ys = torch.linspace(1, h - 2, grid_dim, device=device).long()
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
            grid_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            num_grid = grid_coords.shape[0]
            if num_grid >= patches_per_image:
                selected_coords = grid_coords[:patches_per_image]
            else:
                extra = patches_per_image - num_grid
                rand_x = torch.randint(1, w - 1, (extra,), device=device)
                rand_y = torch.randint(1, h - 1, (extra,), device=device)
                extra_coords = torch.stack([rand_x, rand_y], dim=-1)
                selected_coords = torch.cat([grid_coords, extra_coords], dim=0)
            x = selected_coords[:, 0].unsqueeze(0).repeat(n, 1)
            y = selected_coords[:, 1].unsqueeze(0).repeat(n, 1)

        elif centroid_sel_strat == 'GREEDY':
            # Açgözlü algoritma: Yüksek gradyan değerine sahip ve yamaların örtüşmesini engelleyecek şekilde seçim yapılır.
            # __image_gradient fonksiyonu, görüntü gradyanlarını hesaplar.
            quality = self.__image_gradient(images)[0]  # Beklenen şekil: (n, h, w)
            selected_x = []
            selected_y = []
            for i in range(n):
                q = quality[i]  # (h, w) kalite haritası
                q_flat = q.flatten()
                sorted_indices = torch.argsort(q_flat, descending=True)
                selected = []
                for idx in sorted_indices:
                    y_coord = (idx // w).item()
                    x_coord = (idx % w).item()
                    # Sınır kontrolü: kenar bölgeleri hariç tutulur.
                    if x_coord < 1 or x_coord > w - 2 or y_coord < 1 or y_coord > h - 2:
                        continue
                    valid = True
                    # Mevcut seçilen yamalarla çakışma kontrolü: Aynı yamaların birbirine yakın olmaması için patch boyutu kullanılır.
                    for (sel_x, sel_y) in selected:
                        if abs(x_coord - sel_x) < P and abs(y_coord - sel_y) < P:
                            valid = False
                            break
                    if valid:
                        selected.append((x_coord, y_coord))
                    if len(selected) >= patches_per_image:
                        break
                # Yeterli yama bulunamazsa, eksik kısım rastgele seçilir.
                if len(selected) < patches_per_image:
                    extra = patches_per_image - len(selected)
                    extra_x = torch.randint(1, w - 1, (extra,), device=device)
                    extra_y = torch.randint(1, h - 1, (extra,), device=device)
                    selected.extend(list(zip(extra_x.tolist(), extra_y.tolist())))
                selected = selected[:patches_per_image]
                sel_tensor = torch.tensor(selected, device=device)
                selected_x.append(sel_tensor[:, 0])
                selected_y.append(sel_tensor[:, 1])
            x = torch.stack(selected_x, dim=0)
            y = torch.stack(selected_y, dim=0)

        elif centroid_sel_strat == 'RANDOM':
            x = torch.randint(1, w - 1, size=[n, patches_per_image], device=device)
            y = torch.randint(1, h - 1, size=[n, patches_per_image], device=device)

        else:
            raise NotImplementedError(f"Yama merkezi seçimi uygulanmadı: {centroid_sel_strat}")

        coords = torch.stack([x, y], dim=-1).float()
        imap_patches = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        gmap_patches = altcorr.patchify(fmap[0], coords, P // 2).view(b, -1, 128, P, P)

        if return_color:
            clr = altcorr.patchify(images[0], 4 * (coords + 0.5), 0).view(b, -1, 3)

        if disps is None:
            disps = torch.ones(b, n, h, w, device=device)

        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(grid[0], coords, P // 2).view(b, -1, 3, P, P)

        index = torch.arange(n, device=device).view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)

        if return_color:
            return fmap, gmap_patches, imap_patches, patches, index, clr
        return fmap, gmap_patches, imap_patches, patches, index

    def __compute_edge_map(self, images):
        """
        Sobel operatörü kullanarak kenar haritası hesaplar.

        Args:
            images: Giriş görüntü tensörü. Beklenen boyut: [batch, channel, height, width]
                    Eğer tensör [batch, 1, channel, height, width] şeklinde geliyorsa, ekstra boyut
                    kaldırılır.
        Returns:
            Kenar haritası tensörü.
        """
        # Eğer görüntü tensörü 5 boyutlu ise ve ikinci boyut 1 ise, bu boyutu kaldırıyoruz.
        if images.dim() == 5 and images.size(1) == 1:
            images = images.squeeze(1)

        # Renkli görüntüyü gri tonlamaya çeviriyoruz.
        gray = images.mean(dim=1, keepdim=True)

        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=gray.dtype, device=gray.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=gray.dtype, device=gray.device).unsqueeze(0).unsqueeze(0)

        grad_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)

        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return edge_map


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout)]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):
    def __init__(self, use_viewer=False):
        super(VONet, self).__init__()
        self.P = 3
        self.patchify = Patchifier(self.P)
        self.update = Update(self.P)

        self.DIM = DIM
        self.RES = 4


    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False):
        """ Estimates SE3 or Sim3 between pair of frames """

        images = 2 * (images / 255.0) - 0.5
        intrinsics = intrinsics / 4.0
        disps = disps[:, :, 1::4, 1::4].float()

        fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps)

        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p//2, p//2]
        patches = set_depth(patches, torch.rand_like(d))

        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"), indexing='ij')
        ii = ix[kk]

        imap = imap.view(b, -1, DIM)
        net = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)
        
        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"), indexing='ij')
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"), indexing='ij')

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))

        return traj

