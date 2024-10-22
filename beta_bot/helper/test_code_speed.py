import datetime


def test_code_speed(start: datetime, end: datetime):
    print(f'{(end-start).microseconds * 0.001} ms')
