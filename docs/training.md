# Подбор параметров и прежняя оценка CSV

Из корня проекта, после установки окружений по README:

```bash
export CUDA_VISIBLE_DEVICES=0
cp configs/train.example.json configs/train.local.json
```

В `train.local.json` укажите `input_csv`, `image_root` и соответствие `columns`. Пути к CSV и корню изображений считаются от папки плана. Названия окружений уже настроены на 3 + 1. Для локальных checkpoints можно заменить идентификаторы в `models` абсолютными путями.

В обучающем CSV нужны кандидат, эталон, путь к изображению и число клинически значимых ошибок (`target`). В репозитории есть `russian_RadEvalX.csv`, но снимки нужно предоставить отдельно.

Проверить команды, вычислить признаки, затем обучить смесь:

```bash
conda run -n cxr-fit python build_method.py --plan configs/train.local.json --work-dir work/train --dry-run
conda run --no-capture-output -n cxr-fit python build_method.py --plan configs/train.local.json --work-dir work/train --features-only --resume
conda run -n cxr-fit python build_method.py --plan configs/train.local.json --work-dir work/train --fit-only
```

Можно убрать `--features-only` и выполнить расчёт признаков и fit одной командой. Fit использует nested CV, выбирает перевод, текстовую метрику, веса смеси, нормализацию и логистическую калибровку. Веса нейросетей не обучаются. Результаты:

- `work/train/features.csv` и отдельные файлы этапов;
- `quality_method_config.json` — полная конфигурация с provenance и результатами CV;
- `quality_components.json` — выбранные компоненты для нового JSON-интерфейса;
- `quality_weights.json` — только scaling, fusion, calibration для слияния.

Если задан `--output-json`, два компактных JSON записываются рядом с ним. Для оценки с ними укажите `score_component.py --config /path/quality_components.json` и `merge_scores.py --weights /path/quality_weights.json`.

Преобразование RadCliQ по умолчанию — `reciprocal`, требующее положительных значений. Если значения неположительны, код завершится с ошибкой. Выбор `training.lower_transform="negate"` — явное изменение методики; он не включается автоматически. При смене только параметров fit признаки пересчитывать не нужно.

Прежняя оценка нового CSV без экспертного `target`:

```bash
conda run --no-capture-output -n cxr-fit python evaluate.py --config quality_method_config.json --input new_reports.csv --work-dir work/inference --output quality_scores.json --resume
```

Этот интерфейс сохраняет прежний формат `summary` / `entry-wise` с `quality_score`, выбранными метриками и GREEN-объяснениями. Полный артефакт проверяет совпадение кода с обучением. Для изменения корня изображений или окружений используется `--execution-config configs/execution.example.json`; номер GPU задавайте через ENV.

Для ранее обученного артефакта можно экспортировать компактные файлы командой `python export_weights.py quality_method_config.json`. Новый JSON-интерфейс использует те же формулы и выбранные модели; полный старый CLI сохраняет прежнюю строгую проверку версии кода.

Служебный скрипт разовой миграции GREEN-кэша перенесён из `tools/` в `scripts/migrations/revalidate_green_hotfix_cache.py`; он не нужен для обычного запуска. Миграция разрешена только при изменении одного GREEN-файла и не подходит для обновления всего репозитория.
