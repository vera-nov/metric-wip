# Запуск на сервере с одной NVIDIA GPU 40 ГБ

Инструкция относится к обновлённому архиву этой рабочей копии. Изменения не отправлялись в исходный GitHub: один `git clone` исходного репозитория не добавит `pipeline.py` и новые конфигурации. Распакуйте новый архив поверх свежего клона. Предполагаются Linux, установленная conda, интернет, NVIDIA GPU и доступ к нужным данным.

## Какие окружения объединены

| Имя | Назначение | Зависимости |
|---|---|---|
| `cxr-fit` | Управление заданиями, nested CV, подбор весов и калибровки | `requirements-fit.txt` |
| `cxr-qwen` | Перевод Qwen3.5-9B и русский GREEN | `requirements-qwen.txt` |
| `cxr-translate` | Оба остальных переводчика, HY и TranslateGemma, последовательно | `requirements-translation.txt` |
| `cxr-text` | Все семь английских текстовых метрик | `requirements-text.txt` |
| `cxr-image` | Визуальная BioViL-T и её текстовая часть | `requirements-biovilt.txt` |

Предыдущее разделение RadGraph, RadCliQ, RaTEScore и энкодеров на разные окружения было консервативным. Прямые требования RadEval 0.2.0 и RadGraph 0.1.18 допускают Transformers 4.46.3. Все семь текстовых оценок можно направить в один `cxr-text`. Pin 4.30.2 для BioViL-T в предыдущем варианте не был доказанным обязательным ограничением модели. Полное выполнение объединённого стека на GPU ещё не проверено, поэтому перед полным расчётом ниже предусмотрен smoke test.

Визуальный стек оставлен отдельным для проверки старого hi-ml; его также можно будет объединить с `cxr-text` после проверки совместимости. Русский GREEN и Qwen-перевод остаются отдельно: RadEval 0.2.0 требует Transformers <5, а для Qwen3.5 используется современная ветка 5.x. Установка библиотек нескольких моделей в одном окружении не загружает их веса в память.

`cxr-fit` необязателен как отдельное окружение: команды управления и fit можно запускать из `cxr-text`, поскольку там уже есть базовые зависимости. Ниже лёгкое окружение оставлено для независимого CPU-этапа подбора параметров.

## 1 Подготовить код

Вы уже склонировали репозиторий. Из его корня:

```bash
unzip -o /absolute/path/to/russian-cxr-report-evaluation-updated.zip -d ..
test -f pipeline.py
conda --version
nvidia-smi
```

Архив содержит верхнюю папку `russian-cxr-report-evaluation`. Команда предполагает, что свежий клон имеет это стандартное имя. Если клон назван иначе, распакуйте архив отдельно и скопируйте его содержимое в клон.

## 2 Создать окружения и установить пакеты

Все команды выполняются из корня проекта. Контроллер:

```bash
conda create -n cxr-fit python=3.11 pip -y
conda run -n cxr-fit python -m pip install -r requirements-fit.txt
```

Четыре GPU-окружения:

```bash
for env_name in cxr-qwen cxr-translate cxr-text cxr-image; do
  conda create -n "$env_name" python=3.11 pip -y || break
  conda run -n "$env_name" python -m pip install \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124 || break
done
```

Это конкретный пример для CUDA 12.4 wheels с совместимым драйвером. Если драйвер сервера не поддерживает эту сборку, используйте другую официально согласованную CUDA-сборку PyTorch. Не нужно устанавливать отдельный CUDA toolkit только ради готовых wheels. Успешный `nvidia-smi` сам по себе не доказывает работоспособность установленного PyTorch: ниже есть проверка CUDA из Python.

```bash
conda run -n cxr-qwen python -m pip install \
  -c constraints-gpu-cu124.txt -r requirements-qwen.txt

conda run -n cxr-translate python -m pip install \
  -c constraints-gpu-cu124.txt -r requirements-translation.txt

conda run -n cxr-text python -m pip install \
  -c constraints-gpu-cu124.txt -r requirements-text.txt

conda run -n cxr-image python -m pip install \
  -c constraints-gpu-cu124.txt -r requirements-biovilt.txt
```

Requirements — стартовые спецификации, не GPU-проверенные lock-файлы. Если resolver сообщает конфликт, остановитесь на нём и сохраните сообщение; не обходите его через `--no-deps`. После каждой команды установки проверьте её успешное завершение.

```bash
for env_name in cxr-qwen cxr-translate cxr-text cxr-image; do
  conda run -n "$env_name" python -m pip check || break
  conda run -n "$env_name" python -c \
    'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), "BF16:", torch.cuda.is_bf16_supported())' || break
done
```

`pip check` проверяет метаданные зависимостей, но не доказывает, что remote-код моделей работает. Для этого нужен расчёт на небольшом числе реальных примеров.

## 3 Настроить доступ к весам

На странице `google/translategemma-12b-it` под своим аккаунтом Hugging Face примите условия модели. Затем выполните интерактивный вход на сервере. Совместимый с обеими ветками huggingface_hub способ:

```bash
conda run --no-capture-output -n cxr-qwen python -c \
  'from huggingface_hub import login; login()'
```

Вход выполняется под тем же пользователем ОС, который запускает остальные окружения; стандартное хранилище авторизации у них общее. Не записывайте токен в план или команды. Если на сервере уже настроена авторизация Hugging Face, повторный вход не нужен. Веса загружаются при первом запуске этапа и затем берутся из локального кэша. Можно заранее скачать snapshots и прописать их абсолютные пути в `models`.

## 4 Подготовить данные и план

В репозитории есть CSV с 100 отчётами, но нет изображений. Разместите снимки на сервере и проверьте, что `image_root + path_to_image` указывает на существующий файл. Если CSV содержит `RadEvalX_images/13/CXR13_IM-0198-1001.png`, а снимок лежит в `/data/cxr/RadEvalX_images/13/...`, задайте `image_root=/data/cxr`, а не `/data/cxr/RadEvalX_images`.

```bash
cp configs/train.server40gb.example.json configs/train.server.json
```

В `configs/train.server.json` задайте:

- `input_csv`: абсолютный путь к обучающему CSV либо путь относительно JSON;
- `image_root`: корень для относительных путей изображений;
- `runtime.gpu`: номер выделенной GPU; при Slurm или уже заданной `CUDA_VISIBLE_DEVICES` лучше `null`, чтобы сохранить внешнее ограничение;
- `columns`: свои названия столбцов, если отличаются от примера.

Для A100 40 ГБ стартовые размеры: GREEN 1, перевод 1 (зашит как последовательный обход), текстовые энкодеры 8, изображения 1. Код выбирает BF16 на поддерживающей его GPU, иначе FP16. Это режим инференса базовых моделей, не их дообучение.

Оставлено `training.lower_transform="reciprocal"`, как в статье. Если RadCliQ окажется неположительным, fit остановится. Если вы выбираете знаковое преобразование `-RadCliQ`, явно поменяйте это поле на `negate`; этот выбор меняет метод относительно текущего текста статьи. Не меняйте его автоматически, чтобы получить желаемую корреляцию.

Проверка всех путей и плана:

```bash
conda run -n cxr-fit python -c '
from pathlib import Path
from pipeline import load_plan, resolved_plan, prepare
plan = resolved_plan(load_plan("configs/train.server.json"))
base, _ = prepare(plan, "work/preflight", True)
missing = [x for x in base.image_path if not Path(x).is_file()]
assert not missing, missing[:10]
print("Rows:", len(base), "all image paths exist")
'

conda run -n cxr-fit python build_method.py \
  --plan configs/train.server.json --work-dir work/train --dry-run

conda run -n cxr-fit python -m unittest discover -s tests -v
```

## 5 Выполнить пробный нейросетевой расчёт

Сначала два примера через все реальные модели — без fit и без изменения числа фолдов полного эксперимента:

```bash
conda run -n cxr-fit python -c '
from pathlib import Path
import pandas as pd
from pipeline import load_plan, atomic_json
plan = load_plan("configs/train.server.json")
id_col = plan["columns"].get("row_id", "report_id")
small = pd.read_csv(plan["input_csv"], dtype={id_col: str}, keep_default_na=False).head(2)
Path("work").mkdir(exist_ok=True)
small.to_csv("work/smoke.csv", index=False)
plan["input_csv"] = str(Path("work/smoke.csv").resolve())
atomic_json("configs/smoke.server.json", plan)
'

conda run --no-capture-output -n cxr-fit python build_method.py \
  --plan configs/smoke.server.json --work-dir work/smoke \
  --features-only --resume
```

В другом терминале можно наблюдать память через `watch -n 1 nvidia-smi`. Новая версия также записывает `gpu_memory` в `*.versions.json`: это максимумы allocated/reserved, измеренные аллокатором PyTorch внутри конкретного worker, а не общая память GPU со всеми чужими процессами. Два примера подтверждают загрузку и работу интерфейсов, но не являются измерением пика для самого длинного отчёта полного датасета.

## 6 Посчитать полные признаки

```bash
conda run --no-capture-output -n cxr-fit python build_method.py \
  --plan configs/train.server.json --work-dir work/train \
  --features-only --resume
```

Одна эта команда последовательно выполнит:

1. Qwen `terms`, Qwen `noterms`, HY `terms`, HY `noterms`, TranslateGemma `noterms`;
2. русский GREEN;
3. для каждого варианта перевода — RadGraph, RadCliQ, RaTEScore, оценки CXR-BERT, оценки BioViL-T и image-text BioViL-T.

Внутри одной настройки перевода кандидаты и эталоны обрабатываются одной загруженной моделью, по одному тексту. Затем worker завершается. Контроллер ждёт его завершения перед следующим заданием. Одновременно три переводчика не загружаются. Текстовый и визуальный компоненты повторно используют тот же CSV перевода.

Выход: `work/train/features.csv` и отдельные CSV/метаданные этапов. При сбое повторите ту же команду: `--resume` переиспользует подходящие завершённые этапы. Не запускайте несколько контроллеров одновременно на этой GPU и в одном `work-dir`.

Можно запускать только нужные задания с автоматическими зависимостями. Например, все переводы:

```bash
conda run --no-capture-output -n cxr-fit python pipeline.py run \
  --plan configs/train.server.json --work-dir work/train --resume \
  --only translate.qwen.terms,translate.qwen.noterms,translate.hy_mt.terms,translate.hy_mt.noterms,translate.translategemma.noterms
```

После отдельных заданий основная команда с `--features-only --resume` достроит оставшиеся и соберёт сводный CSV.

## 7 Подобрать параметры метрики

```bash
conda run --no-capture-output -n cxr-fit python build_method.py \
  --plan configs/train.server.json --work-dir work/train \
  --fit-only --output-json quality_method_config.json
```

Это CPU-этап: nested CV, выбор конфигурации, весов, нормализации и калибровки. Веса больших нейросетей не обучаются. На выходе — `quality_method_config.json` с выбранными компонентами, параметрами и результатами внешних фолдов. Если выбранный режим reciprocal отклонит RadCliQ, изучите признаки и примите описанное выше решение о преобразовании, затем повторите только fit. Менять только параметры CV или lower_transform можно без пересчёта признаков.

## 8 Посчитать метрику на новых отчётах

Нужны новый CSV с кандидатом, эталоном, изображением и ID; экспертное число ошибок не требуется. Названия столбцов берутся из обученного плана.

```bash
conda run --no-capture-output -n cxr-fit python evaluate.py \
  --config quality_method_config.json \
  --input /absolute/path/to/new_reports.csv \
  --work-dir work/inference --output quality_scores.json --resume
```

Если у новых изображений другой корень, передайте `--execution-config configs/execution.example.json`, предварительно указав правильный `image_root` в этом JSON. Вычисляются один выбранный перевод, русский GREEN, одна текстовая метрика и визуальный компонент. Параметры смеси остаются обученными.

После успешного пробного расчёта и до полного эксперимента закрепите рабочие версии окружений:

```bash
mkdir -p locks
for env_name in cxr-fit cxr-qwen cxr-translate cxr-text cxr-image; do
  conda env export -n "$env_name" > "locks/$env_name.yml"
  conda run -n "$env_name" python -m pip freeze > "locks/$env_name.pip.txt"
done
```

## Оценка памяти

Грубая нижняя оценка только для BF16/FP16-весов — два байта на параметр: Qwen 9B около 18 ГБ, HY 7B около 14 ГБ, TranslateGemma 12B около 24 ГБ, в десятичных единицах. Это расчёт по порядку числа параметров, не измерение реального потребления. Дополнительно нужны KV-cache, временные тензоры, CUDA-контекст и запас аллокатора. Размеры самих checkpoints и визуальных частей могут отличаться от округлённого имени модели.

При коротких рентгенологических отчётах, генерации по одному тексту и свободной NVIDIA GPU 40 ГБ ожидается, что каждая из этих моделей поместится без квантования. Самая тесная — TranslateGemma 12B. Гарантировать отсутствие OOM без запуска на конкретной карте и самых длинных входах нельзя. Длинные входы, большой batch и другие GPU-процессы меняют оценку. Если 40 ГБ относятся к системной RAM, а не VRAM, по этому числу сделать вывод о размещении моделей на GPU невозможно.

При OOM сначала проверьте посторонние GPU-процессы, число одновременно запущенных заданий и batch. Для перевода batch уже равен 1. Снижение max_new_tokens уменьшает потенциальный cache, но обрезанные ответы здесь отклоняются, поэтому лимит нельзя уменьшать вслепую. Квантование и замена моделей меняют компонент и требуют нового расчёта признаков и обучения смеси; это не настройка, которую следует переключать между обучением и инференсом незаметно.
