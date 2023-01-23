__all__ = ['Autocomplete']


class Autocomplete:
    """
    Add in python file all the available functions and classes in a  list
    EG. __all__ = ['DecisionTreeClassifier','RandomForestClassifier']
    """

    def __init__(self, file_path: str = __file__):
        self.file_path = file_path
        self.name = '__all__'

    def run(self):
        self.generate_func_names()
        self.generate_class_names()

    def generate_func_names(self):
        self.name = "__functions__"
        new_line, lines = self.autofind("def ")
        self.autowrite(new_line, lines)

    def generate_class_names(self):
        self.name = "__classes__"
        new_line, lines = self.autofind(pattern="class ")
        self.autowrite(new_line, lines)

    def autofind(self, pattern: str):
        """ find string and save to file at file_path in list named name"""
        with open(self.file_path, "r") as f:
            lines = f.readlines()

        selection = [l for l in lines if pattern in l]
        functions = []

        for line in selection:
            if pattern in line[:len(pattern)]:  # has def at the beggining of line
                func_name = line.split(pattern)[1]
                func_name = func_name.split("(")[0]
                func_name = func_name.split(":")[0]
                functions.append(func_name)
        new_line = f"{self.name} = {functions}"

        return new_line, lines

    def autowrite(self, new_line, lines):

        with open(self.file_path, "w") as f:
            f.write(new_line)
            f.write("\n")
            for line in lines:
                if self.name in line[:len(self.name)]:
                    continue
                f.write(line)
        print(f"Successfully saved at {self.file_path} in {self.name} list")


if __name__ == "__main__":
    Autocomplete().run()
