# Зависимости и проверка совместимости

Четыре requirements-файла соответствуют четырём окружениям. `scripts/setup_envs.sh` ставит согласованную пару torch/torchvision напрямую; отдельный `constraints-gpu-cu124.txt` удалён. Генеративные модели используют один стек Transformers 5.x, текстовые метрики — 4.46.3, BioViL-T image — 4.30.2. Окружения отделяют несовместимые версии Transformers; модели загружаются последовательно отдельными процессами.

## spaCy 3.8.2 в cxr-text

Одного добавления `spacy==3.8.2` в старый Python 3.11 недостаточно:

- [RadEval 0.2.0](https://pypi.org/pypi/radeval/0.2.0/json) требует `numpy<2` и medspaCy.
- [spaCy 3.8.2](https://pypi.org/pypi/spacy/3.8.2/json) требует `thinc>=8.3.0,<8.4.0`.
- [Thinc 8.3.2](https://pypi.org/pypi/thinc/8.3.2/json) требует NumPy 2, а [Thinc 8.3.3](https://pypi.org/pypi/thinc/8.3.3/json) допускает NumPy 1.26.
- [medspaCy 1.3.1](https://pypi.org/pypi/medspacy/1.3.1/json) требует spaCy <3.8 на Python <3.12 и допускает spaCy >=3.8,<4 на Python >=3.12.

Поэтому `cxr-text`: **Python 3.12, numpy 1.26.4, spaCy 3.8.2, Thinc 8.3.3, medspaCy 1.3.1**. spaCy CUDA extras не устанавливаются; GPU для нейросетевых метрик использует PyTorch. Остальные окружения сохраняют Python 3.11.

## Что проверять

При установке скрипт запускает `pip check`. Для отдельной проверки разрешения зависимостей в Python 3.12:

```bash
python -m pip install --dry-run --ignore-installed -r requirements-text.txt --report text-resolution.json
```

Это проверка метаданных, не выполнение моделей. Реальную совместимость загрузчиков, checkpoints, CUDA и конкретного снимка проверяет `smoke_test.py --image /path/to/cxr.png`. Он запускает 10 этапов: три перевода, GREEN, пять групп текстовых метрик (семь оценок) и image/text BioViL-T. В `*.versions.json` этапов сохраняются версии пакетов и данные памяти GPU.

Unit-тесты проверяют сохранение входных полей, соответствие записей A/B/C, численное совпадение с прежней формулой, кэширование, изоляцию стадий и прежние build/evaluate. Нейронные worker-процессы в этих тестах подменены; PASS unit-тестов не означает PASS реальных моделей.

Проверено 26 октября 2026: 27 unit-тестов прошли; разрешение зависимостей всех трёх модельных окружений прошло (Python 3.11 для generation/image, 3.12 для text). Связка NumPy / spaCy / Thinc / medspaCy дополнительно установлена в изолированное окружение: импорты, разбиение короткого текста на предложения и `pip check` прошли. GPU и checkpoints в среде разработки отсутствуют, поэтому реальный модельный smoke test здесь не выполнялся.

После успешного серверного smoke test можно сохранить точный состав окружений:

```bash
mkdir -p locks
for env_name in cxr-fit cxr-qwen cxr-text cxr-image; do
  conda env export -n "$env_name" > "locks/$env_name.yml"
  conda run -n "$env_name" python -m pip freeze > "locks/$env_name.pip.txt"
done
```
