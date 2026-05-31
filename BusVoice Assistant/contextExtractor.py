import os
from datetime import datetime
from pathlib import Path

def collect_project_context(output_filename=None):
    """
    Собирает содержимое всех файлов проекта в один текстовый файл.
    Ищет файлы с расширениями: .py, .txt, .csv, .log
    """
    
    # Определяем директорию, где находится скрипт
    script_dir = Path(__file__).parent
    
    # Создаем имя выходного файла с временной меткой
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"project_context_{timestamp}.txt"
    
    output_path = script_dir / output_filename
    
    # Расширения файлов для включения
    target_extensions = {'.py', '.txt', '.csv', '.log'}
    
    # Директории и файлы для игнорирования
    ignore_dirs = {
        '.git', '__pycache__', '.venv', 'venv', 'env', 
        '.env', 'node_modules', '.idea', '.vscode',
        'build', 'dist', '.eggs', '*.egg-info'
    }
    
    ignore_files = {
        output_filename,  # Игнорируем сам выходной файл
        '.DS_Store',
        '*.pyc',
        '*.pyo',
        '*.pyd'
    }
    
    collected_files = []
    total_size = 0
    
    print(f"🔍 Сканирую директорию: {script_dir}")
    print(f"📁 Ищу файлы: {', '.join(target_extensions)}")
    print("-" * 60)
    
    # Собираем все подходящие файлы
    for root, dirs, files in os.walk(script_dir):
        # Фильтруем директории для игнорирования
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        # Конвертируем путь в относительный от script_dir
        rel_root = Path(root).relative_to(script_dir)
        
        for file in files:
            file_path = Path(root) / file
            file_extension = file_path.suffix.lower()
            
            # Проверяем расширение
            if file_extension in target_extensions:
                # Проверяем, не игнорируем ли файл
                if file_path.name != output_filename:
                    rel_path = file_path.relative_to(script_dir)
                    file_size = file_path.stat().st_size
                    
                    # Пропускаем слишком большие файлы (>10MB)
                    if file_size > 10 * 1024 * 1024:
                        print(f"⚠️  Пропущен большой файл: {rel_path} ({file_size:,} bytes)")
                        continue
                    
                    collected_files.append((file_path, rel_path, file_size))
                    total_size += file_size
    
    if not collected_files:
        print("❌ Не найдено подходящих файлов для выгрузки")
        return None
    
    print(f"📊 Найдено файлов: {len(collected_files)}")
    print(f"📦 Общий размер: {total_size:,} bytes")
    print(f"💾 Сохраняю в: {output_path}")
    print("-" * 60)
    
    # Записываем содержимое в выходной файл
    try:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            # Записываем заголовок
            output_file.write("=" * 80 + "\n")
            output_file.write(f"КОНТЕКСТ ПРОЕКТА\n")
            output_file.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            output_file.write(f"Корневая директория: {script_dir}\n")
            output_file.write(f"Количество файлов: {len(collected_files)}\n")
            output_file.write(f"Общий размер: {total_size:,} bytes\n")
            output_file.write("=" * 80 + "\n\n")
            
            # Записываем оглавление
            output_file.write("ОГЛАВЛЕНИЕ:\n")
            output_file.write("-" * 80 + "\n")
            for i, (file_path, rel_path, file_size) in enumerate(collected_files, 1):
                output_file.write(f"{i:3d}. {rel_path} ({file_size:,} bytes)\n")
            output_file.write("=" * 80 + "\n\n")
            
            # Записываем содержимое каждого файла
            for i, (file_path, rel_path, file_size) in enumerate(collected_files, 1):
                print(f"📄 [{i}/{len(collected_files)}] Читаю: {rel_path}")
                
                output_file.write("\n" + "=" * 80 + "\n")
                output_file.write(f"ФАЙЛ {i}: {rel_path}\n")
                output_file.write(f"Размер: {file_size:,} bytes\n")
                output_file.write(f"Тип: {file_path.suffix}\n")
                output_file.write("-" * 80 + "\n\n")
                
                try:
                    # Пытаемся прочитать как текстовый файл
                    with open(file_path, 'r', encoding='utf-8') as input_file:
                        content = input_file.read()
                        output_file.write(content)
                        
                        # Добавляем перенос строки в конце, если его нет
                        if content and not content.endswith('\n'):
                            output_file.write('\n')
                            
                except UnicodeDecodeError:
                    # Если не получается прочитать как UTF-8, пробуем другие кодировки
                    try:
                        with open(file_path, 'r', encoding='latin-1') as input_file:
                            content = input_file.read()
                            output_file.write(content)
                            if content and not content.endswith('\n'):
                                output_file.write('\n')
                    except Exception as e:
                        output_file.write(f"[ОШИБКА ЧТЕНИЯ ФАЙЛА: {str(e)}]\n")
                        print(f"  ⚠️  Ошибка чтения: {rel_path} - {str(e)}")
                except Exception as e:
                    output_file.write(f"[ОШИБКА ОБРАБОТКИ ФАЙЛА: {str(e)}]\n")
                    print(f"  ⚠️  Ошибка обработки: {rel_path} - {str(e)}")
                
                output_file.write("\n" + "=" * 80 + "\n")
        
        print("-" * 60)
        print(f"✅ Успешно создан файл: {output_path}")
        print(f"📊 Статистика:")
        print(f"   - Файлов обработано: {len(collected_files)}")
        print(f"   - Общий размер: {total_size:,} bytes")
        print(f"   - Расширения: {', '.join(sorted(set(f.suffix.lower() for _, f, _ in collected_files)))}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Ошибка при создании файла: {str(e)}")
        return None

def main():
    """
    Основная функция
    """
    print("🚀 СБОРЩИК КОНТЕКСТА ПРОЕКТА")
    print("=" * 60)
    
    # Собираем проект
    result = collect_project_context()
    
    if result:
        print("\n📋 Файл готов для отправки в Telegram бот!")
        print("💡 Вы можете отправить файл как документ или скопировать его содержимое")
        
        # Показываем размер для проверки лимитов Telegram
        file_size = os.path.getsize(result)
        max_telegram_size = 50 * 1024 * 1024  # 50 MB для ботов
        
        if file_size > max_telegram_size:
            print(f"⚠️  Внимание! Размер файла ({file_size:,} bytes) превышает лимит Telegram (50 MB)")
            print("💡 Рекомендую разбить на части или исключить некоторые файлы")
        else:
            print(f"✅ Размер файла ({file_size:,} bytes) в пределах лимита Telegram")
    else:
        print("\n❌ Не удалось создать файл с контекстом проекта")

if __name__ == "__main__":
    main()