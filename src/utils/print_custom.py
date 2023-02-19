__all__ = ['printBeautyTable']

from prettytable import PrettyTable


def printBeautyTable(table, col_name: list = ["Variable", "Name"]):
    """
    print parallel lists/arrays, neatly, in columns with beautiful structure

    Format:
    +----------+-------------------------+
    | Variable |           Name          |
    +----------+-------------------------+
    |    RF    |      Random Forest      |
    |   MLP    | Multiple Neural Network |
    +----------+-------------------------+
    """

    tbl = PrettyTable(col_name)

    for i in range(len(table)):
        tbl.add_row(table[i])

    print(tbl)


# -----------------------------------------------------------
if __name__ == "__main__":
    variable = ["RF",
                "DT",
                "LR",
                "GB",
                "LB",
                "XB",
                "MLP"]

    names = ["Random Forest",
             "Decision Tree",
             "Logistic Regression",
             "Gradient Boost",
             "Light Boost",
             "XG Boost",
             "Multiple Neural Network"]

    print('Information for variables:')
    list_concat = [list(zipped) for zipped in zip(variable, names)]  # combine list
    printBeautyTable(table=list_concat, col_name=["Variable", "Name"])
