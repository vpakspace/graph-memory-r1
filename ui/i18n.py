"""Internationalization (i18n) for Graph-Memory-R1 Streamlit UI."""

from __future__ import annotations

from typing import Callable

TRANSLATIONS: dict[str, dict[str, str]] = {
    # App
    "app_title": {"en": "Graph-Memory-R1", "ru": "Graph-Memory-R1"},
    "app_subtitle": {
        "en": "GRPO-trained Memory Manager with Graph Memory (Neo4j)",
        "ru": "GRPO-обученный Memory Manager с графовой памятью (Neo4j)",
    },
    "language": {"en": "Language", "ru": "Язык"},

    # Tabs
    "tab_chat": {"en": "Chat", "ru": "Чат"},
    "tab_graph": {"en": "Graph Explorer", "ru": "Граф"},
    "tab_training": {"en": "Training", "ru": "Обучение"},
    "tab_benchmark": {"en": "Benchmark", "ru": "Benchmark"},

    # Chat tab
    "chat_header": {"en": "Conversation & Memory", "ru": "Разговор и память"},
    "chat_input": {"en": "Enter conversation text", "ru": "Введите текст разговора"},
    "chat_placeholder": {
        "en": "User: I live in Paris and I love Python programming.",
        "ru": "User: Я живу в Париже и люблю программирование на Python.",
    },
    "chat_process": {"en": "Process & Update Memory", "ru": "Обработать и обновить память"},
    "chat_processing": {"en": "Processing conversation...", "ru": "Обработка разговора..."},
    "chat_operations": {"en": "Memory Operations", "ru": "Операции с памятью"},
    "chat_op_count": {
        "en": "{total} operations ({ok} successful, {fail} failed)",
        "ru": "{total} операций ({ok} успешных, {fail} неудачных)",
    },
    "chat_question": {"en": "Ask a question about the conversation", "ru": "Задайте вопрос о разговоре"},
    "chat_question_placeholder": {"en": "Where does the user live?", "ru": "Где живёт пользователь?"},
    "chat_ask": {"en": "Ask", "ru": "Спросить"},
    "chat_asking": {"en": "Generating answer...", "ru": "Генерация ответа..."},
    "chat_answer": {"en": "Answer", "ru": "Ответ"},
    "chat_sources": {"en": "Sources ({count})", "ru": "Источники ({count})"},

    # Graph tab
    "graph_header": {"en": "Graph Memory Explorer", "ru": "Обозреватель графовой памяти"},
    "graph_stats": {"en": "Graph Statistics", "ru": "Статистика графа"},
    "graph_nodes": {"en": "Total nodes: {count}", "ru": "Всего узлов: {count}"},
    "graph_edges": {"en": "Total edges: {count}", "ru": "Всего связей: {count}"},
    "graph_core": {"en": "Core Memory", "ru": "Ядро памяти"},
    "graph_semantic": {"en": "Semantic Nodes", "ru": "Семантические узлы"},
    "graph_episodic": {"en": "Эпизодические узлы", "ru": "Эпизодические узлы"},
    "graph_semantic_en": {"en": "Semantic Nodes", "ru": "Семантические узлы"},
    "graph_episodic_en": {"en": "Episodic Nodes", "ru": "Эпизодические узлы"},
    "graph_clear": {"en": "Clear Graph Memory", "ru": "Очистить память графа"},
    "graph_clear_confirm": {"en": "Type DELETE to confirm", "ru": "Введите DELETE для подтверждения"},
    "graph_cleared": {"en": "Graph memory cleared", "ru": "Память графа очищена"},

    # Training tab
    "train_header": {"en": "GRPO Training", "ru": "GRPO обучение"},
    "train_status": {"en": "Training Status", "ru": "Статус обучения"},
    "train_not_started": {"en": "Training not started", "ru": "Обучение не начато"},
    "train_dataset": {"en": "Dataset Path", "ru": "Путь к датасету"},
    "train_epochs": {"en": "Epochs", "ru": "Эпохи"},
    "train_lr": {"en": "Learning Rate", "ru": "Скорость обучения"},
    "train_start": {"en": "Start Training", "ru": "Начать обучение"},
    "train_running": {"en": "Training in progress...", "ru": "Обучение в процессе..."},
    "train_complete": {"en": "Training complete!", "ru": "Обучение завершено!"},
    "train_loss": {"en": "Final Loss: {loss}", "ru": "Итоговый Loss: {loss}"},
    "train_lora": {"en": "LoRA saved: {path}", "ru": "LoRA сохранён: {path}"},
    "train_load_lora": {"en": "Load LoRA Weights", "ru": "Загрузить LoRA веса"},
    "train_lora_path": {"en": "LoRA adapter path", "ru": "Путь к LoRA адаптеру"},
    "train_lora_loaded": {"en": "LoRA weights loaded", "ru": "LoRA веса загружены"},

    # Benchmark tab
    "bench_header": {"en": "LoCoMo Benchmark", "ru": "LoCoMo Benchmark"},
    "bench_dataset": {"en": "Dataset Path", "ru": "Путь к датасету"},
    "bench_max_items": {"en": "Max items to evaluate", "ru": "Максимум вопросов"},
    "bench_use_judge": {"en": "Use LLM-as-Judge", "ru": "Использовать LLM-as-Judge"},
    "bench_run": {"en": "Run Benchmark", "ru": "Запустить Benchmark"},
    "bench_running": {"en": "Running benchmark ({current}/{total})...", "ru": "Запуск benchmark ({current}/{total})..."},
    "bench_results": {"en": "Benchmark Results", "ru": "Результаты Benchmark"},
    "bench_avg_f1": {"en": "Average F1: {f1:.3f}", "ru": "Средний F1: {f1:.3f}"},
    "bench_avg_em": {"en": "Average EM: {em:.3f}", "ru": "Средний EM: {em:.3f}"},
    "bench_avg_bleu": {"en": "Average BLEU-1: {bleu:.3f}", "ru": "Средний BLEU-1: {bleu:.3f}"},
    "bench_col_q": {"en": "Q#", "ru": "Q#"},
    "bench_col_category": {"en": "Category", "ru": "Категория"},
    "bench_col_f1": {"en": "F1", "ru": "F1"},
    "bench_col_question": {"en": "Question", "ru": "Вопрос"},

    # Common
    "error": {"en": "Error: {msg}", "ru": "Ошибка: {msg}"},
    "no_data": {"en": "No data available", "ru": "Данные отсутствуют"},
}


def get_translator(lang: str = "en") -> Callable[..., str]:
    """Return a translator function t(key, **kwargs) for the given language."""

    def t(key: str, **kwargs) -> str:
        entry = TRANSLATIONS.get(key)
        if entry is None:
            return key
        text = entry.get(lang, entry.get("en", key))
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, IndexError):
                return text
        return text

    return t
