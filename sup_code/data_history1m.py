import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('btc_data_download.log')
    ]
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.bybit.com/v5/market/kline"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
CATEGORY = "linear"
START_DATE = datetime(2025, 4, 14)
END_DATE = datetime.now()


async def fetch_data(session, start_ts, end_ts):
    params = {
        'category': CATEGORY,
        'symbol': SYMBOL,
        'interval': '1',
        'start': str(int(start_ts.timestamp() * 1000)),
        'end': str(int(end_ts.timestamp() * 1000)),
        'limit': 1000
    }

    logger.info(f"Запрашиваю данные с {start_ts} по {end_ts}")

    try:
        async with session.get(BASE_URL, params=params) as response:
            result = await response.json()
            if result['retCode'] != 0 or 'list' not in result['result']:
                error_msg = f"API Error: {result.get('retMsg', 'Unknown error')}"
                logger.error(error_msg)
                raise Exception(error_msg)

            records_count = len(result['result']['list'])
            logger.info(f"Получено {records_count} записей за период {start_ts} - {end_ts}")
            return result['result']['list']

    except Exception as e:
        logger.error(f"Ошибка при запросе данных: {str(e)}")
        raise


async def fetch_all_data():
    logger.info(f"Начало загрузки данных с {START_DATE} по {END_DATE}")

    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        current_start = START_DATE
        batch_num = 1

        while current_start < END_DATE:
            current_end = current_start + timedelta(minutes=999)
            if current_end > END_DATE:
                current_end = END_DATE

            logger.debug(f"Пакет {batch_num}: {current_start} - {current_end}")
            tasks.append(fetch_data(session, current_start, current_end))
            current_start = current_end + timedelta(minutes=1)
            batch_num += 1

        logger.info(f"Всего пакетов для загрузки: {len(tasks)}")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results


def process_data(raw_data):
    logger.info("Начало обработки полученных данных")
    all_data = []
    error_count = 0

    for i, batch in enumerate(raw_data, 1):
        if isinstance(batch, Exception):
            logger.warning(f"Ошибка в пакете {i}: {str(batch)}")
            error_count += 1
            continue
        all_data.extend(batch)

    logger.info(f"Обработано пакетов: {len(raw_data)}, ошибок: {error_count}")

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close',
        'volume', 'turnover'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')

    logger.info(f"Всего записей после обработки: {len(df)}")
    return df.sort_values('timestamp').drop_duplicates()


async def main():
    try:
        logger.info("=== Начало работы программы ===")
        start_time = datetime.now()

        raw_data = await fetch_all_data()
        df = process_data(raw_data)

        filename = f"btc_{INTERVAL}_from_{START_DATE.date()}.csv"
        df.to_csv(filename, index=False, header=False)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Сохранено {len(df)} записей в файл {filename}")
        logger.info(f"Общее время выполнения: {duration:.2f} секунд")
        logger.info("=== Работа программы завершена успешно ===")

    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Ошибка в основном потоке: {str(e)}")