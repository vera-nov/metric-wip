# Оценка русскоязычных рентгенологических отчетов

## Порядок запуска полного пайплайна

Пайплайн состоит из трёх основных этапов: вычисление признаков → подбор оптимальной конфигурации → оценка новых данных.

### 1. Подготовка окружения и данных

```
git clone https://github.com/vera-nov/metric-wip.git
cd metric-wip

conda create -n cxr-fit python=3.11 pip -y
conda run -n cxr-fit python -m pip install -r requirements-fit.txt
```

Для нейросетевых этапов дополнительно нужны окружения `cxr-qwen`, `cxr-translate`, `cxr-text`, `cxr-image`. Инструкции по их установке, включая PyTorch и CUDA, находятся в [SERVER_RUNBOOK_RU.md](https://github.com/vera-nov/metric-wip/blob/main/SERVER_RUNBOOK_RU.md).

Создать рабочую конфигурацию:

```
cp configs/train.server40gb.example.json configs/train.local.json
```

В `configs/train.local.json` указать:

- `input_csv` — путь к обучающей таблице;
- `image_root` — корневую директорию рентгеновских снимков;
- `runtime.gpu` — номер используемой GPU;
- `columns` — соответствие названий столбцов.

В обучающем CSV необходимы кандидат (`candidate`), эталон (`reference`), путь к снимку (`image`) и число клинически значимых ошибок (`target`).

Важно: снимков в репозитории нет. Также нужен доступ Hugging Face к модели TranslateGemma.

### 2. Вычисление признаков

Сначала проверить план запуска:

```
conda run -n cxr-fit python build_method.py \
  --plan configs/train.local.json \
  --work-dir work/train \
  --dry-run
```

Затем запустить вычисления:

```
conda run -n cxr-fit python build_method.py \
  --plan configs/train.local.json \
  --work-dir work/train \
  --features-only --resume
```

На этом этапе последовательно вычисляются:

1. Переводы RU→EN с использованием Qwen, HY-MT и TranslateGemma.
2. Русская метрика GREEN.
3. Английские текстовые метрики (RadGraph, RadCliQ, RaTEScore, BERTScore, cosine similarity).
4. Визуальная метрика на основе BioViL-T.

Результат: `work/train/features.csv` и отдельные CSV для каждого компонента.

Флаг `--resume` позволяет продолжить расчёт после сбоя без повторного вычисления успешно завершённых этапов.

### 3. Подбор оптимальной конфигурации

После вычисления всех признаков выполнить:

```
conda run -n cxr-fit python build_method.py \
  --plan configs/train.local.json \
  --work-dir work/train \
  --fit-only \
  --output-json quality_method_config.json
```

На этом этапе используется nested cross-validation (5×5) для выбора:

- оптимальной комбинации текстовых и визуальных метрик;
- весов трёх компонентов;
- параметров нормализации и логистической калибровки.

Результат: `quality_method_config.json` — итоговая конфигурация метрики, необходимая для дальнейшей оценки. На этом этапе GPU не требуется.

Примечание: если RadCliQ выдаёт нулевые или отрицательные значения, режим `lower_transform: "reciprocal"` вызовет ошибку. Можно явно выбрать `"negate"`, но это изменение методики преобразования признака.

### 4. Оценка на новых данных

Подготовить `new_reports.csv` с теми же названиями столбцов для кандидата, эталона и изображения, что и при обучении. Экспертная разметка ошибок (`target`) уже не требуется.

Запустить:

```
conda run -n cxr-fit python evaluate.py \
  --config quality_method_config.json \
  --input new_reports.csv \
  --work-dir work/inference \
  --output quality_scores.json \
  --resume
```

Результат: `quality_scores.json`, содержащий итоговую оценку `quality_score` в диапазоне [0, 1] для каждого отчёта, оценки выбранных компонентов и объяснения GREEN.

При оценке используются только компоненты и параметры, выбранные на этапе обучения. Повторный подбор конфигурации не выполняется.
