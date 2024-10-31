import serial
import time

# Arduino'nun bağlı olduğu seri portu belirleyin (Windows için 'COM3', Linux için '/dev/ttyUSB0' olabilir)
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  # Seri portu başlat

time.sleep(2)  # Arduino'nun başlatılması için biraz bekle


def send_command(command):
    ser.write(command.encode())  # Komutu seri port üzerinden gönder
    print(f"Komut gönderildi: {command}")


while True:
    i = input("Bir şey yazın: ")  # Kullanıcıdan komut al
    # Komut gönder
    send_command(i)
    time.sleep(1)

    if i == '2':  # Kullanıcı '2' girerse programdan çık
        print("Seri port kapatılıyor...")
        ser.close()  # Seri portu kapat
        break
