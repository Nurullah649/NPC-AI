from collections import Counter
import random


def parse_cfg(cfg_rules):
    """CFG'yi (Bağlamdan Bağımsız Dil) kurallarını ayrıştırır."""
    rules = {}
    for rule in cfg_rules:
        left, right = rule.split('-->')
        rules[left.strip()] = [prod.strip() for prod in right.split('|')]
    return rules


def validate_alphabet_and_cfg(alphabet, cfg_rules):
    """Alfabeyi ve CFG kurallarını doğrular."""
    defined_nonterminals = set()
    alphabet.append('|')  # '|' sembolü ayrım için eklenir.

    for rule in cfg_rules:
        if '-->' not in rule:
            raise ValueError("Kural formatı yanlış! '-->' eksik.")

        left, right = rule.split('-->')
        left = left.strip()

        if left not in alphabet and left != 'S':
            raise ValueError(f"Alfabede olmayan bir karakter bulundu: {left}")

        if left.isupper():
            defined_nonterminals.add(left)

        for production in right.split('|'):
            for char in production:
                if char not in alphabet:
                    raise ValueError(f"Alfabede olmayan bir karakter bulundu: {char}")

    # Alfabede tanımlanmış ama CFG'de tanımlanmamış büyük harfleri kontrol et
    for symbol in alphabet:
        if symbol.isupper() and symbol not in defined_nonterminals:
            raise ValueError(f"Alfabede tanımlanmış ama CFG'de tanımlanmamış büyük harf bulundu: {symbol}")


def generate_language_with_duplicates(rules, start_symbol='S'):
    """CFG'ye göre tüm dili üretir ve tekrar eden kelimeleri tespit eder."""
    all_strings = []
    queue = [start_symbol]

    while queue:
        current = queue.pop()
        if all(char.islower() for char in current):  # Tüm karakterler terminalse
            all_strings.append(current)
        else:
            for i, char in enumerate(current):
                if char.isupper() and char in rules:  # Değiştirilebilir bir sembolse
                    for replacement in rules[char]:
                        new_string = current[:i] + replacement + current[i + 1:]
                        queue.append(new_string)

    counter = Counter(all_strings)
    unique_results = sorted(set(all_strings))
    duplicates = sorted([word for word, count in counter.items() if count > 1])
    return unique_results, duplicates


def generate_combinations(unique_language, count=5):
    """Benzersiz kelimelerden rastgele birleşimlerle yeni kelimeler üretir."""
    combinations = set()
    while len(combinations) < count:
        combination = ''.join(random.sample(unique_language, min(2, len(unique_language))))
        combinations.add(combination)
    return sorted(combinations)


def main():
    alphabet = input('Alfabeyi giriniz (örneğin a,b,X): ').split(',')
    cfg_is_valid = False

    while not cfg_is_valid:
        try:
            # Kullanıcıdan CFG kurallarını al
            cfg_input = input("Dili Tanımlayınız (Örn: S-->aa|bX|aXX,X-->ab|b): ").split(',')

            # Alfabeyi ve CFG'yi doğrula
            validate_alphabet_and_cfg(alphabet, cfg_input)
            print("\nAlfabe ve CFG doğrulandı.\n")
            cfg_is_valid = True  # CFG geçerli olduğunda döngüyü kır
        except ValueError as e:
            print(f"Hata: {e}")
            print("Lütfen CFG kurallarını tekrar giriniz.\n")

    # CFG'yi ayrıştır
    rules = parse_cfg(cfg_input)
    print("CFG Kuralları:")
    for left, productions in rules.items():
        print(f"  {left} --> {' | '.join(productions)}")

    # Dil oluştur ve tekrarları tespit et
    unique_language, duplicates = generate_language_with_duplicates(rules)

    # Benzersiz kelimelerden rastgele kombinasyonlar oluştur
    extra_combinations = generate_combinations(unique_language, count=5)

    # Sonuçları yazdır
    print("\nÜretilen Benzersiz Kelimeler:")
    print(", ".join(unique_language))
    if duplicates:
        print("\nTekrarlanan Kelimeler:")
        print(", ".join(duplicates))
        print("\nEkstra Kombinasyonlar:")
        print(", ".join(extra_combinations))
    else:
        print("\nTekrarlanan Kelime Yok.")


if __name__ == "__main__":
    main()
