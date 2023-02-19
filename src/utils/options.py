__all__ = ['Options']


class Options:
    """Class Variables are the Available Options"""

    @classmethod
    def get_string_options(cls) -> list:
        """Return list with the values of global class variables"""
        return sorted([value for name, value in vars(cls).items() if name.isupper()])

    # alternative way
    # def get_string_options(self):
    #     """Return list with the values of global class variables"""
    #     return sorted([ImputationNumericalOptions.__dict__[i] for i in dir(self) if not "__" in i and i.isupper()])

    def get_user_options(self) -> list:
        """Return list with the name of global class variables"""
        return sorted([i for i in dir(self) if not "__" in i and i.isupper()])

    @staticmethod
    def meta_info():
        """Print to explain of what is/means each variable"""
        pass


################################################################################################

class DemoOptions(Options):
    """ Imputation Available Methods for Categoriacal values"""

    # Class variable
    MEAN = "mean"
    MEDIAN = "median"


if __name__ == "__main__":
    print(DemoOptions().get_string_options())
    print(DemoOptions().get_user_options())
