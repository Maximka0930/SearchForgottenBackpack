@echo off
chcp 65001 > nul
echo ========================================
echo 🚀 ЗАПУСК СИСТЕМЫ ДЕТЕКЦИИ ПОТЕРЯННЫХ ВЕЩЕЙ
echo ========================================
echo.

:: Проверяем SSL сертификаты
if exist "ssl_certs\cert.pem" (
    if exist "ssl_certs\key.pem" (
        echo ✅ SSL сертификаты найдены
    ) else (
        echo ❌ Не найден ключ (key.pem)
        goto :no_ssl
    )
) else (
    echo ❌ Не найден сертификат (cert.pem)
    goto :no_ssl
)

:: Получаем IP адрес
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr "IPv4"') do (
    set IP=%%i
    goto :ip_found
)
:ip_found
set IP=%IP: =%

echo.
echo 📡 ДОСТУПНЫЕ АДРЕСА:
echo 💻 Компьютер: https://localhost:5000
echo 📱 Телефон:   https://%IP%:5000
echo.
echo ⚠ ВАЖНО ДЛЯ ТЕЛЕФОНА:
echo 1. Открой браузер на телефоне
echo 2. Введи: https://%IP%:5000
echo 3. Прими предупреждение о сертификате
echo 4. Нажми "Дополнительно" -> "Перейти на сайт"
echo ========================================
echo.

:: Запускаем Flask
python app.py
goto :end

:no_ssl
echo.
echo ❌ SSL сертификаты не найдены!
echo.
echo 📋 Инструкция по созданию:
echo 1. Открой CMD как администратор
echo 2. cd %~dp0
echo 3. mkdir ssl_certs
echo 4. cd ssl_certs
echo 5. openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes
echo.
echo 💡 Пока запускаем без SSL (только локально):
echo    http://localhost:5000
echo ========================================
echo.
python app.py

:end
pause