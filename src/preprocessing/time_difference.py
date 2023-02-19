__all__ = ['TimeDifference', 'Birth']

'''Extract new information from exist datetime data:
   date to age, date distance'''

from datetime import date, datetime


class TimeDifference:

    @staticmethod
    def dates2dayduration(opened_date: datetime, closed_date: datetime) -> int:
        """
        Calculate the duration between opened_date and closed_date
        :param opened_date: the date from start an event
        :param closed_date: the date from end an event
        :return: the duration in format number of days
        """
        date_diff = (closed_date - opened_date)
        date_diff = date_diff.days

        if closed_date < opened_date:
            print('Wrong date time: closed_date > opened_date')
            return date_diff * -1
        else:
            return date_diff


class Birth:

    # def calculate_age(born):
    #     today = date.today()
    #     return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    @staticmethod
    def date2age(born: datetime):

        """
        Calculate age from birthdate.

        url: https://stackoverflow.com/questions/2217488/age-from-birthdate-in-python

        :param born: birth date
        :return:
        """

        today = date.today()
        try:
            birthday = born.replace(year=today.year)
        except ValueError:  # raised when birth date is February 29 and the current year is not a leap year
            birthday = born.replace(year=today.year, month=born.month + 1, day=1)
        if birthday > today:
            return today.year - born.year - 1
        else:
            return today.year - born.year

    @staticmethod
    def age2date(age):
        """
        Calculate birthdate from age.

        :param age: the number of age
        :return:
        """
        # TODO age2date
        pass

    @staticmethod
    def death_age():
        # TODO find the age of death
        pass

    @staticmethod
    def distance_between_two_dates():
        # TODO find how many years between to ages
        pass
