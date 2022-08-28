import numpy as np, faiss, torch, scipy.stats
from faiss import normalize_L2
import time
import torch.nn.functional as F

def update_plabels(database, X, k=50, max_iter=20, IsCALP=False):
    print('Updating pseudo-labels...')
    alpha = 0.99
    labels = np.asarray(database.all_labels)
    labeled_idx = np.asarray(database.labeled_idx)
    unlabeled_idx = np.asarray(database.unlabeled_idx)
    num_class = database.num_class
    d = X.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatIP(res, d, flat_config)
    normalize_L2(X)
    index.add(X)
    N = X.shape[0]
    c = time.time()
    D, I = index.search(X, k + 1)
    elapsed = time.time() - c
    print('kNN Search done in %d seconds' % elapsed)
    D = D[:, 1:] ** 3
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1.0 / np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D
    if IsCALP == True:
        Z = CALP(num_class, Wn, alpha, labeled_idx, labels, N, max_iter)
    else:
        Z = LP(num_class, Wn, alpha, labeled_idx, labels, N, max_iter)
    probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
    probs_l1[probs_l1 < 0] = 0
    entropy = scipy.stats.entropy(probs_l1.T)
    weights = 1 - entropy / np.log(len(database.classes))
    weights = weights / np.max(weights)
    p_labels = np.argmax(probs_l1, 1)
    if IsCALP == True:
        p_labels, weights = remove_Anchors(num_class, p_labels, weights)
    correct_idx = p_labels == labels
    acc = correct_idx.mean()
    p_labels[labeled_idx] = labels[labeled_idx]
    weights[labeled_idx] = 1.0
    database.p_weights = weights.tolist()
    database.p_labels = p_labels
    for i in range(len(database.classes)):
        cur_idx = np.where(np.asarray(database.p_labels) == i)[0]
        database.class_weights[i] = float(labels.shape[0]) / len(database.classes) / cur_idx.size

    return (
     acc, weights.tolist(), p_labels)


def CALP(num_class, Wn, alpha, labeled_idx, labels, N, max_iter):
    Z = np.zeros((N, num_class))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(num_class):
        cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
        y = np.zeros((N,))
        y[cur_idx] = 1.0 / cur_idx.shape[0]
        y[-2 * num_class + i] = 1.0 / cur_idx.shape[0]
        y[-num_class + i] = -1.0 / cur_idx.shape[0]
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-06, maxiter=max_iter)
        Z[:, i] = f

    Z[Z < 0] = 0
    return Z


def LP(num_class, Wn, alpha, labeled_idx, labels, N, max_iter):
    Z = np.zeros((N, num_class))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(num_class):
        cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
        y = np.zeros((N,))
        y[cur_idx] = 1.0 / cur_idx.shape[0]
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-06, maxiter=max_iter)
        Z[:, i] = f

    Z[Z < 0] = 0
    return Z


def Expand_Positive_Negtive_anchor(batch_features, p_labels, w, num_class):
    r = 10 * num_class
    codesize = list(batch_features.shape)
    codesize[0] = 0
    anchor_y = np.array([]).reshape(codesize)
    anchor_n = np.array([]).reshape(codesize)
    p_labels = np.array(p_labels)
    w = np.array(w)
    for i in range(num_class):
        min_index = np.where(p_labels == i)[0][np.argsort(w[np.where(p_labels == i)[0]])[:r]]
        max_index = np.where(p_labels == i)[0][np.argsort(w[np.where(p_labels == i)[0]])[-r:]]
        Extend_batch_features_y = np.dot(w[min_index], batch_features[min_index]) / np.sum(w[min_index])
        Extend_batch_features_n = np.dot(w[max_index], batch_features[max_index]) / np.sum(w[max_index])
        anchor_y = np.vstack([anchor_y, Extend_batch_features_y])
        anchor_n = np.vstack([anchor_n, Extend_batch_features_n])

    Expand_batch_features = np.vstack([batch_features, anchor_y, anchor_n]).astype(np.float32)
    return (
     Expand_batch_features, anchor_y, anchor_n)


def remove_Anchors(num_class, p_labels, uncertainly_weights):
    p_labels = p_labels[:-2 * num_class]
    uncertainly_weights = uncertainly_weights[:-2 * num_class]
    return (
     p_labels, uncertainly_weights)

