#Polish Notasyonu
def calculate(operand,numbers):
    result=0.0
    match operand:
        case '*':
            result = numbers[0] * numbers[1]
        case '+':
            result = numbers[0] + numbers[1]
        case '/':
            if numbers[1]!=0:
                result = numbers[0] / numbers[1]
            else:
                print("0'a bölünme hatası sonuç 0 dönecektir")
        case '-':
            result = numbers[0] - numbers[1]
    return result

prefix_denklem = input('Polish bir denklem giriniz: ')
prefix_denklem = prefix_denklem.split()
operands = ['+', '*', '-', '/']

for i in range(len(prefix_denklem)):
    if prefix_denklem[i] not in operands:
        try:
            prefix_denklem[i] = int(prefix_denklem[i])
        except ValueError:
            print(f"Hatalı giriş tespit edildi: {prefix_denklem[i]}")
            raise ValueError("Geçersiz operand veya operatör!")
while len(prefix_denklem)!=1:
    for i in range(len(prefix_denklem)):
        if prefix_denklem[i] in operands and (prefix_denklem[i+1] not in operands and prefix_denklem[i+2] not in operands):
            result=calculate(prefix_denklem[i],[int(prefix_denklem[i+1]),int(prefix_denklem[i+2])])
            prefix_denklem[i]=result
            prefix_denklem.pop(i+1)
            prefix_denklem.pop(i+1)
            break
print(prefix_denklem)