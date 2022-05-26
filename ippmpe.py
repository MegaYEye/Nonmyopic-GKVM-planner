from gplib import GPR
from tsp_solver.greedy import solve_tsp
import numpy as np

def MPE_k(gpr, mesh, k):
    X_current, y_current = gpr.X.cpu().numpy(), gpr.y.cpu().numpy()
    X_origin = X_current
    y_origin = y_current
    i=0
    preds = []
    while i<k:
        pred, uncertainty = gpr.test(mesh.astype(np.float)) 
        # print(np.max(uncertainty))
        pos = mesh[np.argmax(uncertainty)]
        preds.append(pos)
        X_next = np.vstack([X_current, pos[None,:]])
        y_next = np.append(y_current,pred[np.argmax(uncertainty)])
        
        gpr.set_observation(X_next, y_next) 
        X_current, y_current = X_next, y_next
        i+=1
    gpr.set_observation(X_origin, y_origin)
    return np.array(preds)

def MPE_thres(gpr, mesh, thres):
    X_current, y_current = gpr.X.cpu().numpy(), gpr.y.cpu().numpy()
    X_origin = X_current
    y_origin = y_current
    i=0
    preds = []
    while True:
        pred, uncertainty = gpr.test(mesh.astype(np.float)) 
        pos = mesh[np.argmax(uncertainty)]
        preds.append(pos)
        X_next = np.vstack([X_current, pos[None,:]])
        y_next = np.append(y_current,pred[np.argmax(uncertainty)])
        
        gpr.set_observation(X_next, y_next) 
        X_current, y_current = X_next, y_next
        i+=1
        if np.max(uncertainty)<thres:
            break
    gpr.set_observation(X_origin, y_origin)
    return np.array(preds)


def create_gpr(X, y, config):
    gpr = GPR(X,y, config)
    return gpr

def norm(x):
    return np.sqrt(np.sum(x**2))

def TSP(landmarks):
    D = np.zeros((len(landmarks), len(landmarks)))
    for i in range(len(landmarks)):
        for j in range(len(landmarks)):
            D[i,j] = norm(landmarks[i]-landmarks[j])
    path = solve_tsp( D, endpoints=(0,None))
    return path

    