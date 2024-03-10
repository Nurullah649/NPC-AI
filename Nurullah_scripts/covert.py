# -*- coding: utf-8 -*-

from xml.dom import minidom
import os
import glob
from colorama import Fore, Back, Style


lut = {}
lut["motor"]=0
lut["tasit"] = 0
lut["arac"]=0
lut["tren"]=0
lut["ismak"]=0
lut["bis"]=0
lut["1"]=0
lut["2"]=0
lut["3"]=0
lut["4"]=0
lut["5"]=0
lut["car"]=0
lut["kam"]=0
lut["mot"]=0
lut["bus"]=0
lut["insan"] = 1
lut["yaya"]=1
lut["ins"]=1
lut["uuap"]=2
lut["yuap"] = 2
lut["yaup"]=2
lut["uyam"] = 3
lut["yaam"] = 3




def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_xml2yolo(lut):
    for fname in glob.glob("/home/nurullah/Masaüstü/traffic_birdseye/*.xml"):

        xmldoc = minidom.parse(fname)

        #fname_out = (fname[:-4] + '.txt')
        fname_out = ("/home/nurullah/Masaüstü/deneme_txt_konum/" + os.path.basename(fname)[:-4] + '.txt')

        with open(fname_out, "w") as f:

            itemlist = xmldoc.getElementsByTagName('object')
            size = xmldoc.getElementsByTagName('size')[0]
            width = int((size.getElementsByTagName('width')[0]).firstChild.data)
            height = int((size.getElementsByTagName('height')[0]).firstChild.data)

            for item in itemlist:
                # get class label
                classid = (item.getElementsByTagName('name')[0]).firstChild.data
                if classid in lut:
                    label_str = str(lut[classid])
                else:
                    label_str = "-1"
                    print(Fore.RED+"warning: label "+fname+" görselde '%s' not in look-up table" %classid+Style.RESET_ALL)

                # get bbox coordinates
                xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                if(width==0):
                    print(Fore.RED + "warning:" + fname + " görselde sorun var" + Style.RESET_ALL)

                bb = convert_coordinates((width, height), b)
                # print(bb)

                f.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')

        print("wrote %s" % fname_out)


def main():
    convert_xml2yolo(lut)


if __name__ == '__main__':
    main()