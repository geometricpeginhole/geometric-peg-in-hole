import pybullet as p
import pybullet_planning as pp
import numpy as np

class RealsenseCamera():
    def __init__(self):
        self.width = 640
        self.height = 480
        self.view_matrix = p.computeViewMatrix([0, 0, 0], [0, 0, -1], [0, 1, 0])
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=45, aspect=self.width / self.height, nearVal=0.1, farVal=100.0)
    
    def get_obs(self):
        rgb = np.array(p.getCameraImage(width=self.width, height=self.height,
                                        viewMatrix=self.view_matrix,
                                        projectionMatrix=self.projection_matrix,
                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])
        return {'rgb': rgb}

    def set_pose(self, T):
        self.view_matrix = p.computeViewMatrix(cameraEyePosition=T[:3, 3],
                                               cameraTargetPosition=T[:3, 3] - T[:3, 2],
                                               cameraUpVector=T[:3, 1])
        # self.view_matrix = T

if __name__ == '__main__':
    camera = RealsenseCamera()
    proj = np.array(RealsenseCamera().projection_matrix).reshape(4, 4)
    view = np.array(RealsenseCamera().view_matrix).reshape(4, 4)
    print(proj)
    print(view)