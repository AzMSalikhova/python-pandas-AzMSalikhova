import pandas as pd
from datetime import (
    date,
)
from typing import (
    Optional,
)


class MedicalDeviceAnalyzer:
    """Анализирует данные медицинских устройств из Excel-файла."""

    STATUS_MAP = {
        'operational': 'operational',
        'op': 'operational',
        'ok': 'operational',
        'working': 'operational',
        'planned_installation': 'planned_installation',
        'to_install': 'planned_installation',
        'planned': 'planned_installation',
        'scheduled_install': 'planned_installation',
        'maintenance_scheduled': 'maintenance_scheduled',
        'maint_sched': 'maintenance_scheduled',
        'service_scheduled': 'maintenance_scheduled',
        'maintenance': 'maintenance_scheduled',
        'faulty': 'faulty',
        'broken': 'faulty',
        'error': 'faulty',
        'needs_repair': 'faulty',
    }

    DATE_COLUMNS = ['install_date', 'warranty_until', 'last_calibration_date', 'last_service_date']

    def __init__(self) -> None:
        """Инициализирует анализатор медицинских устройств.

        Attributes:
            df (Optional[pd.DataFrame]): DataFrame с загруженными данными.
        """

        self.df: Optional[pd.DataFrame] = None

    # Приватный метод(для методов этого же класса)
    def _format_date_series(self, series: pd.Series) -> pd.Series:
        """Форматирует даты в строки формата ГГГГ-ММ-ДД.

        Args:
            series: Серия pandas с объектами date.

        Returns:
            Серия pandas со строками дат или None для пропущенных значений.
        """

        # notna - возвращает True, если есть дата, False - если пусто
        # apply - применить к каждому элементу series(столбца) и положить значение в ячейку
        return series.apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)

    def load_data(self, file_path: str) -> None:
        """Загружает данные из Excel-файла.

        Args:
            file_path: Путь к Excel-файлу с данными.
        """

        self.df = pd.read_excel(file_path)
        # Удалить лишние пробелы из названий столбцов, предварительно превратить объект columns в строку
        # это важно - далее мы будем обращаться к столбцам через их названия
        self.df.columns = self.df.columns.str.strip()

    def normalize_status(self) -> None:
        """Приводит столбец статуса к 4 категориям. Неизвестные значения помечает как 'unknown'."""

        # Привести к нижнему регистру и удалить пробелы
        # первый str делает все ячейки строками, второй и третий str открывает доступ к стр. методам для каждой ячейки
        status_series = self.df['status'].astype(str).str.lower().str.strip()
        self.df['status_normalized'] = status_series.map(self.STATUS_MAP)

    def parse_dates(self) -> None:
        """Преобразует столбцы с датами в формат date. При ошибках парсинга ставит NaT."""

        for col in self.DATE_COLUMNS:
            # errors функция поставит специальное значение NaT (Not a Time), которое pandas понимает как "пропущ. дата"
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce', dayfirst=True, format='mixed')
            # Затем извлекаем только дату(без времени)
            self.df[col] = self.df[col].dt.date

    def filter_by_warranty(self, current_date: Optional[date] = None) -> pd.DataFrame:
        """Фильтрует устройства с действующей гарантией.

        Args:
            current_date: Дата для проверки гарантии.

        Returns:
            DataFrame с устройствами, у которых warranty_until >= current_date.
        """

        # Когда дата для фильтрации не выбрана, берем сегодняшнюю
        if current_date is None:
            current_date = date.today()

        warranty = self.df['warranty_until']
        # notna() - булева маска, возвращает True для существующих данных и False для NaN/None/NaT
        not_null = warranty.notna()
        # Возвращаем таблицу, где дата не пуста И гарантия соблюдается
        valid_mask = not_null & warranty.apply(lambda x: x >= current_date)

        return self.df[valid_mask].copy()

    def top_clinics_by_problems(self, n: int = 10) -> pd.DataFrame:
        """Возвращает топ клиник с наибольшим количеством проблем.

        Args:
            n: Количество клиник для вывода (по умолчанию 10).

        Returns:
            DataFrame с колонками clinic_id, clinic_name, total_issues,
            отсортированный по убыванию проблем.
        """

        # Группируем каждую клинику отдельно и суммируем все его ошибки
        grouped = self.df.groupby(['clinic_id', 'clinic_name'], as_index=False).agg(
            total_issues=('issues_reported_12mo', 'sum')
        )
        top = grouped.sort_values('total_issues', ascending=False).head(n)

        return top

    def calibration_report(self, days_threshold: int = 365) -> pd.DataFrame:
        """Формирует отчет по просроченной калибровке.

        Args:
            days_threshold: Порог дней для просрочки (по умолчанию 365).

        Returns:
            DataFrame с устройствами, у которых калибровка просрочена или отсутствует,
            включая колонки days_since_calibration и overdue.
        """

        today = date.today()
        cal = self.df['last_calibration_date'].copy()

        # Для расчёта разницы в днях используем apply с date объектами
        days_since = cal.apply(lambda x: (today - x).days if pd.notna(x) else None)

        # Пометить как просроченные, если превышен порог или дата отсутствует
        overdue = (cal.isna()) | (days_since > days_threshold)

        # Построить отчёт
        report = self.df[['device_id', 'clinic_id', 'clinic_name', 'model', 'last_calibration_date']].copy()
        report['days_since_calibration'] = days_since
        report['overdue'] = overdue
        report = report.sort_values('days_since_calibration', ascending=False, na_position='last')

        return report

    def pivot_summary(self) -> pd.DataFrame:
        """Создает сводную таблицу по клиникам и моделям(только основные показатели).

        Returns:
            DataFrame со сводной информацией:
            - строки: названия клиник
            - столбцы: модели устройств с суффиксами _count, _issues, _uptime
            - значения: количество устройств, сумма проблем, средний аптайм
        """

        # Подготовить копию, заполнив пропуски в числовых столбцах
        df_pivot = self.df.copy()
        df_pivot['issues_reported_12mo'] = df_pivot['issues_reported_12mo'].fillna(0)
        df_pivot['uptime_pct'] = df_pivot['uptime_pct'].fillna(0)

        # Количество устройств
        count_pivot = pd.pivot_table(
            df_pivot,
            index='clinic_name',
            columns='model',
            values='device_id',
            aggfunc='count',
            fill_value=0
        )

        # Сумма проблем
        issues_pivot = pd.pivot_table(
            df_pivot,
            index='clinic_name',
            columns='model',
            values='issues_reported_12mo',
            aggfunc='sum',
            fill_value=0
        )

        # Средний аптайм
        uptime_pivot = pd.pivot_table(
            df_pivot,
            index='clinic_name',
            columns='model',
            values='uptime_pct',
            aggfunc='mean',
            fill_value=0
        )

        # Объединить в один DataFrame с многоуровневыми столбцами
        result = pd.concat(
            [count_pivot.add_suffix('_count'),
             issues_pivot.add_suffix('_issues'),
             uptime_pivot.add_suffix('_uptime')],
            axis=1
        )

        return result


def main():
    # Создать экземпляр анализатора
    analyzer = MedicalDeviceAnalyzer()

    analyzer.load_data('medical_diagnostic_devices_10000.xlsx')

    analyzer.normalize_status()
    analyzer.parse_dates()
    print("- данные загружены, даты и статусы обработаны")

    # Выполнить все расчеты
    valid_warranty = analyzer.filter_by_warranty()
    top_clinics = analyzer.top_clinics_by_problems(n=10)
    cal_report = analyzer.calibration_report(days_threshold=365)
    pivot = analyzer.pivot_summary()

    # Подготовить данные для Excel
    valid_warranty_export = valid_warranty.copy()
    valid_warranty_export['warranty_until'] = analyzer._format_date_series(valid_warranty_export['warranty_until'])

    cal_report_export = cal_report.copy()
    cal_report_export['last_calibration_date'] = analyzer._format_date_series(
        cal_report_export['last_calibration_date'])

    # Сохранить результаты в Excel с сообщениями о каждом листе
    with pd.ExcelWriter('device_analysis_report.xlsx') as writer:
        valid_warranty_export.to_excel(writer, sheet_name='valid_warranty', index=False)
        print("- данные отфильтрованы по гарантии")

        top_clinics.to_excel(writer, sheet_name='top_clinics_problems', index=False)
        print("- клиники с наибольшим кол-вом проблем найдены")

        cal_report_export.to_excel(writer, sheet_name='calibration_report', index=False)
        print("- отчет по срокам калибровки создан")

        pivot.to_excel(writer, sheet_name='pivot_summary', index=False)
        print("- сводная таблица сделана")

    print("\nВсе отчёты сохранены в файл 'device_analysis_report.xlsx'.")


if __name__ == "__main__":
    main()
