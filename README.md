# Оценка рентгенологических отчётов

Три независимые компоненты: русский GREEN, текстовая метрика после перевода RU→EN, BioViL-T image/text. Отдельный CPU-скрипт применяет обученные scaling → fusion → calibration и сохраняет итог.

## Установка окружений

Нужны Linux, conda, доступ к Hugging Face и подходящий NVIDIA-драйвер.

```bash
git clone https://github.com/vera-nov/metric-wip.git
cd metric-wip
bash scripts/setup_envs.sh
export CUDA_VISIBLE_DEVICES=0
```

| Окружение | Python | Назначение |
|---|---|---|
| `cxr-qwen` | 3.11 | GREEN, Qwen, HY-MT, TranslateGemma — последовательно |
| `cxr-text` | 3.12 | Все семь английских текстовых метрик; spaCy 3.8.2 |
| `cxr-image` | 3.11 | BioViL-T image/text |
| `cxr-fit` | 3.11 | Контроллер, подбор параметров и CPU-слияние |

Скрипт установки проверяет `pip check` и останавливается при ошибке. По умолчанию используются PyTorch 2.6.0 / torchvision 0.21.0 CUDA 12.4. Для другой совместимой сборки задайте `TORCH_INDEX_URL` перед установкой. GPU выбирается только через `CUDA_VISIBLE_DEVICES`; номер GPU в JSON не нужен. Установку выполняют один раз в новые окружения.

Для TranslateGemma заранее примите условия `google/translategemma-12b-it` и авторизуйтесь:

```bash
conda run --no-capture-output -n cxr-qwen python -c 'from huggingface_hub import login; login()'
```

## Проверка каждой модели

Перед большим расчётом выполните один короткий пример с настоящим рентгеновским снимком:

```bash
conda run --no-capture-output -n cxr-fit python smoke_test.py --image /data/cxr/example.png
```

Проверяются три переводчика, GREEN, RadGraph, RadCliQ, RaTEScore, CXR-BERT, BioViL-T text и image/text. Модели запускаются последовательно; отчёт — `work/smoke/smoke_result.json`. Ошибка любого этапа завершает команду с ненулевым кодом. `--resume` продолжает успешные этапы; `--dry-run` показывает команды без загрузки моделей.

## Оценка

`reports.json` — JSON-список должен иметь вид:

```json
[
  {
    "report_id": "0001",
    "reference_text": "Лёгкие без очаговых изменений. Выпота нет.",
    "candidate_text": "Очаговых изменений и выпота не выявлено.",
    "reference_image_path": "/data/cxr/example.png"
  }
]
```

Относительные пути к снимкам считаются от папки входного JSON. Порядок записей и дополнительные поля сохраняются. Есть образец [examples/reports.json](examples/reports.json).

Из корня репозитория запустите:
```bash
bash scripts/evaluate_all.sh
````
или
```bash
conda run --no-capture-output -n cxr-fit python score_component.py russian reports.json
conda run --no-capture-output -n cxr-fit python score_component.py text reports.json
conda run --no-capture-output -n cxr-fit python score_component.py image reports.json
conda run -n cxr-fit python merge_scores.py
```

Получатся `russian_score.json`, `image_score.json`, `text_score.json` и `final_score.json`: каждый — тот же список с единственным добавленным полем `score`. 

Пути при необходимости меняются через `--config`, `--output`, `--work-dir`; у слияния — `--weights` и `--output`. Например:

```bash
conda run -n cxr-fit python merge_scores.py runs/russian.json runs/text.json runs/image.json --output runs/final_scores.json
```

По умолчанию готовые этапы переиспользуются после проверки кэша. `--no-resume` повторно запускает worker-процессы; валидные ответы GREEN из его журнала могут переиспользоваться. Для полностью нового расчёта укажите новый `--work-dir`. Команды с общей рабочей папкой запускайте последовательно.

## Обучение и прежний интерфейс

Краткая инструкция: [docs/training.md](docs/training.md). Прежние аргументы `build_method.py` и `evaluate.py` сохранены.

Локальные тесты без скачивания моделей:

```bash
conda run -n cxr-fit python -m unittest discover -s tests -v
```

Совместимость текстовых зависимостей и границы проверки описаны в [docs/dependencies.md](docs/dependencies.md).
