from numpy import source


class SimulatorManager:

    __instance = None
    @staticmethod
    def getInstance():
        if SimulatorManager.__instance == None:
            SimulatorManager()
        return SimulatorManager.__instance

    def __init__(self, source = None, mic_array = None, fs = 8000) -> None:
        if SimulatorManager.__instance != None:
            raise Exception("SimulatorManager already instantiated")
        else:
            SimulatorManager.__instance = self

        self.src = source
        self.mic = mic_array
        self.fs = fs

    def initialize():
        pass

