import numpy as np

class SoundSource:

    def __init__(
        self,
        position = np.array([[0,0,1]]),
        images = None,
        signal = None,
        directivity = 'o',
    ):
        self.position = position
        self.directivity = directivity
        self.signal = signal
        self.images = (np.array(self.position, dtype = np.float32))
    
    def distance(self, point):
        return np.sqrt(np.sum(np.subtract(self.position, point)**2, axis=1))
    
    def road_reflection(self):
        
        self.images = np.vstack((self.images, np.array([self.position[0][0], self.position[0][1], -self.position[0][2]])))
    
    def add_signal(self, signal):
        self.signal = signal

def imdist(source, point):
    return np.sqrt(np.sum(np.subtract(source.images, point)**2, axis=1))

## TEST
# a = SoundSource(np.array([[0,0,1]]))
# a.road_reflection()

# print('distance with images: ', imdist(a, np.array([[0,0,2]])))