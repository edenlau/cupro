from zoneinfo import ZoneInfo
from datetime import datetime
from src import main_web
import unittest


class MainWebUtilsTest(unittest.TestCase):
    def test_convert_to_hk_time(self):
        utc_time = '2024-01-01T12:00:00+00:00'
        result = main_web.convert_to_hk_time(utc_time)
        hk_tz = ZoneInfo('Asia/Hong_Kong')
        expected = datetime(2024, 1, 1, 20, 0, tzinfo=hk_tz)
        self.assertEqual(result.tzinfo, hk_tz)
        self.assertEqual(result, expected)

    def test_calculate_buzz_volume(self):
        data = [
            {'topic': 'university_hku', 'date_day': '2024-04-01', 'originalUrl': 'a'},
            {'topic': 'university_hku', 'date_day': '2024-04-01', 'originalUrl': 'a'},
            {'topic': 'university_hku', 'date_day': '2024-04-01', 'originalUrl': 'b'},
            {'topic': 'university_cuhk', 'date_day': '2024-04-02', 'originalUrl': 'c'},
        ]
        pivot = main_web.calculate_buzz_volume(data)
        self.assertEqual(pivot['university_hku']['01 Apr'], 2)
        self.assertEqual(pivot['university_cuhk']['02 Apr'], 1)


if __name__ == '__main__':
    unittest.main()
