_constants = {}
_constants_defaults ={
    "c" : 343.0,
    "frac_delay_len": 81
}

class Constants:
    def set(self, name, val):
        _constants[name] = val

    def get(self, name):
        try:
            v = _constants[name]
        except KeyError:
            try:
                v = _constants_defaults[name]
            except:
                raise NameError(name + ": no constant found")
        return v

constants = Constants()