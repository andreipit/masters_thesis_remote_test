
#https://stackoverflow.com/questions/52729346/is-there-a-compact-way-to-declare-several-similar-properties-in-a-python-class
# attribute access customisation hooks 
#https://docs.python.org/3/reference/datamodel.html#customizing-attribute-access

class MyClass(dict):
    def __init__(self, variable1='default1', variable9='default9'):
        self.properties = dict(variable1=variable1, variable9=variable9)
        dict.__init__(self)

    def __getattr__(self, name):
        if name.startswith('variable') and name[8:].isdigit():
            return self.properties.get(name, None)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith('variable') and name[8:].isdigit():
            self.properties[name] = value
            return
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name.startswith('variable') and name[8:].isdigit():
            self.properties[name] = None
            return
        super().__delattr__(name)

if __name__ == '__main__':
    x = MyClass()
    x
    print('sd')
