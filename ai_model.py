import torch
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime

class AdvancedLostItemDetector:
    def __init__(self, model_size='medium', device='cuda'):
        """
        Улучшенный детектор с высокой точностью для камеры
        """
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        print(f"Используется устройство: {self.device}")
        
        # Используем более точные модели
        model_weights = {
            'nano': 'yolov8n.pt',
            'small': 'yolov8s.pt', 
            'medium': 'yolov8m.pt',
            'large': 'yolov8l.pt',
            'xlarge': 'yolov8x.pt'
        }
        
        model_name = model_weights.get(model_size, 'yolov8m.pt')
        print(f"Загружаем модель: {model_name}")
        
        # Загружаем модель
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        # Настройки для детекции
        self.conf_threshold = 0.3
        self.iou_threshold = 0.45
        
        # Классы для детекции
        self.person_class = 0  # person
        self.bag_classes = {
            24: 'suitcase',      # чемодан
            26: 'handbag',       # сумка
            28: 'backpack',      # рюкзак
        }
        
        # Параметры для определения "потерянности"
        self.max_distance = 250  # пикселей до человека
        self.min_time_between_reports = 300  # 5 минут между репортами одного предмета
        self.position_change_threshold = 100  # пикселей для определения сдвига
        
        # Для дедупликации
        self.reported_items = {}  # session_id -> {item_key: last_time}
        self.item_positions = {}  # item_key -> position        

    def analyze_frame(self, frame, session_id, timestamp, output_folder):
        """
        Анализ одного кадра с камеры
        Возвращает: (lost_items, annotated_frame)
        """
        # Детекция объектов
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            half=True if self.device == 'cuda' else False,
            agnostic_nms=True,
            max_det=30,
            classes=list(self.bag_classes.keys()) + [self.person_class]
        )
        
        lost_items = []
        annotated_frame = frame.copy()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Разделяем на людей и сумки
            persons, bags = self._extract_objects(boxes, classes, confidences)
            
            # Находим потерянные предметы
            lost_items = self._find_lost_items(
                frame, bags, persons, session_id, timestamp, output_folder
            )
            
            # Добавляем аннотации
            annotated_frame = self._annotate_frame(annotated_frame, bags, persons, lost_items)
        
        return lost_items, annotated_frame
    
    def _extract_objects(self, boxes, classes, confidences):
        """Извлекает людей и сумки из результатов детекции"""
        persons = []
        bags = []
        
        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            class_id = int(cls)
            # Вычисляем центр прямо здесь
            x1, y1, x2, y2 = box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            if class_id == self.person_class and conf > 0.4:
                persons.append({
                    'id': i,
                    'box': box,
                    'confidence': conf,
                    'center': center
                })
            
            elif class_id in self.bag_classes and conf > self.conf_threshold:
                bags.append({
                    'id': i,
                    'box': box,
                    'class': class_id,
                    'class_name': self.bag_classes[class_id],
                    'confidence': conf,
                    'center': center
                })
        
        return persons, bags
    
    def _find_lost_items(self, frame, bags, persons, session_id, timestamp, output_folder):
        """Находит потерянные предметы с дедупликацией"""
        lost_items = []
        current_time = time.time()
        
        for bag in bags:
            # Проверяем наличие людей рядом
            has_nearby_person = self._check_nearby_persons(bag, persons)
            
            # Если предмет один и уверенность высокая
            if not has_nearby_person and bag['confidence'] > 0.5:
                # Проверяем нужно ли репортить
                should_report = self._should_report_item(
                    session_id, bag, current_time
                )
                
                if should_report:
                    item_info = self._save_and_report_item(
                        frame, bag, session_id, timestamp, output_folder, has_nearby_person
                    )
                    
                    if item_info:
                        lost_items.append(item_info)
                        # Обновляем историю
                        self._update_item_history(session_id, bag)
        
        return lost_items
    
    def _check_nearby_persons(self, bag, persons):
        """Проверяет есть ли люди рядом с сумкой"""
        for person in persons:
            distance = np.linalg.norm(
                np.array(person['center']) - np.array(bag['center'])
            )
            if distance < self.max_distance:
                return True
        return False
    
    def _should_report_item(self, session_id, bag, current_time):
        """Проверяет нужно ли репортить предмет (улучшенная дедупликация)"""
        # Создаем уникальный ключ для предмета: класс + примерная позиция
        pos_x, pos_y = bag['center']
        item_key = f"{bag['class_name']}_{int(pos_x/50)*50}_{int(pos_y/50)*50}"
        
        # Инициализируем для сессии
        if session_id not in self.reported_items:
            self.reported_items[session_id] = {}
        
        session_history = self.reported_items[session_id]
        
        # Если предмет уже зарепорчен
        if item_key in session_history:
            item_history = session_history[item_key]
            time_since_last = current_time - item_history
            
            # Не репортим если прошло меньше 5 минут
            return time_since_last >= self.min_time_between_reports
        
        return True  # Первое обнаружение
    
    def _update_item_history(self, session_id, bag):
        """Обновляет историю предметов"""
        pos_x, pos_y = bag['center']
        item_key = f"{bag['class_name']}_{int(pos_x/50)*50}_{int(pos_y/50)*50}"
        
        if session_id not in self.reported_items:
            self.reported_items[session_id] = {}
        
        self.reported_items[session_id][item_key] = time.time()
    
    def _save_and_report_item(self, frame, bag, session_id, timestamp, output_folder, has_nearby_person):
        """Сохраняет предмет и создает информацию о нем"""
        # Используем полное время в формате МСК
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Создаем имя файла
        item_id = bag['id']
        class_name = bag['class_name']
        confidence = float(bag['confidence'])
        
        # Заменяем : на - в имени файла (для совместимости с файловой системой)
        safe_timestamp = timestamp.replace(':', '-').replace(' ', '_')
        item_filename = f"lost_{session_id}_{safe_timestamp}_{item_id}_{class_name}.jpg"
        item_path = Path(output_folder) / item_filename
        
        # Вырезаем и сохраняем область
        x1, y1, x2, y2 = map(int, bag['box'])
        h, w = frame.shape[:2]
        
        # Добавляем отступы
        padding = 15
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Вырезаем ROI
        bag_roi = frame[y1:y2, x1:x2]
        
        if bag_roi.size > 0:
            # Улучшаем и сохраняем
            bag_roi = self._enhance_image(bag_roi)
            cv2.imwrite(str(item_path), bag_roi)
            
            print(f"⚠ ОБНАРУЖЕН ПОТЕРЯННЫЙ ПРЕДМЕТ!")
            print(f"   Тип: {class_name}")
            print(f"   Уверенность: {confidence:.2f}")
            print(f"   Время: {timestamp}")
            print(f"   Сохранено: {item_filename}")
            print(f"{'-'*40}")
            
            source = 'video' if len(session_id) <= 8 else 'camera'  # ИЛИ наоборот

            # Создаем информацию о предмете
            item_info = {
                'image_url': f'/static/detected/lost_items/{item_filename}',
                'timestamp': timestamp,  # Полное время
                'item_id': item_id,
                'class_name': class_name,
                'confidence': confidence,
                'filename': item_filename,
                'video_name': 'Камера' if source == 'camera' else 'Видео',
                'has_person_nearby': has_nearby_person,
                'position': {
                    'x': float(bag['center'][0]),
                    'y': float(bag['center'][1]),
                    'box': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
                }
            }
            
            return item_info
        
        return None
    
    def _annotate_frame(self, frame, bags, persons, lost_items):
        """Добавляет аннотации к кадру"""
        annotated = frame.copy()
        lost_item_ids = {item['item_id'] for item in lost_items}
        
        # Аннотации для сумок
        for bag in bags:
            x1, y1, x2, y2 = map(int, bag['box'])
            is_lost = bag['id'] in lost_item_ids
            
            # Цвет: красный если потеряна, зеленый если нет
            color = (0, 0, 255) if is_lost else (0, 255, 0)
            thickness = 3 if is_lost else 2
            
            # Рисуем bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Подпись
            label = f"{bag['class_name']} {bag['confidence']:.2f}"
            if is_lost:
                label = f"ПОТЕРЯН: {label}"
            
            # Фон для текста
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness
            )
            cv2.rectangle(
                annotated, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Текст
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness
            )
        
        # Аннотации для людей (тонкие синие)
        for person in persons:
            x1, y1, x2, y2 = map(int, person['box'])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        # Общая информация
        info_text = f"Сумки: {len(bags)} | Люди: {len(persons)} | Потеряно: {len(lost_items)}"
        cv2.putText(
            annotated, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )
        cv2.putText(
            annotated, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1
        )
        
        return annotated
    
    def _enhance_image(self, image):
        """Улучшает качество изображения"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # CLAHE для улучшения контраста
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Резкость
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
        return image
    
    def _analyze_video_file(self, video_path, output_folder, video_id, progress_callback=None, lost_item_callback=None):
        """Внутренний метод для анализа видео файлов"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не могу открыть видео: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lost_items_found = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Анализируем кадр
            lost_items, _ = self.analyze_frame(
                frame, video_id, None, output_folder  # timestamp=None - время генерируется внутри
            )
            
            # Вызываем callback для найденных предметов
            for item in lost_items:
                lost_items_found.append(item)
                if lost_item_callback:
                    lost_item_callback(item)
            
            # Прогресс
            if progress_callback and frame_count % 10 == 0:
                progress = int((frame_count / total_frames) * 100)
                progress_callback(progress)
        
        cap.release()
        
        if progress_callback:
            progress_callback(100)
        
        print(f"\nАнализ видео завершен: {len(lost_items_found)} предметов")
        return lost_items_found
    
    def analyze_video(self, video_path, output_folder, video_id, progress_callback=None, lost_item_callback=None):
        """
        Анализ видео файла (оставлен для обратной совместимости)
        """
        return self._analyze_video_file(video_path, output_folder, video_id, progress_callback, lost_item_callback)
    
    def cleanup_session(self, session_id):
        """Очищает данные сессии"""
        if session_id in self.reported_items:
            del self.reported_items[session_id]

def check_gpu():
    """Проверка доступности GPU"""
    if torch.cuda.is_available():
        print(f"✅ GPU доступен: {torch.cuda.get_device_name(0)}")
        print(f"   Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠ GPU недоступен, используется CPU")
        return False

def analyze_video(video_path, output_folder, video_id, progress_callback=None, lost_item_callback=None):
    """
    Основная функция для анализа видео (для обратной совместимости)
    """
    detector = AdvancedLostItemDetector(model_size='large', device='cuda')
    return detector._analyze_video_file(
        video_path, output_folder, video_id, 
        progress_callback, lost_item_callback
    )

def analyze_single_frame(frame, output_folder, session_id, timestamp):
    """
    Анализ одного кадра с камеры
    """
    detector = AdvancedLostItemDetector(model_size='medium', device='cuda')
    return detector.analyze_frame(frame, session_id, timestamp, output_folder)