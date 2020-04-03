from sklearn.cluster import AffinityPropagation


class AP:
    def __init__(self, x):
        self.ap = AffinityPropagation()
        self.cluster_centers_indices = None
        self.labels = None
        self.x = x

    def fit(self):
        return self.ap.fit(self.x)

    def predict(self):
        self.cluster_centers_indices = self.fit().cluster_centers_indices_
        self.labels = self.ap.labels_
        return self.cluster_centers_indices, self.labels


def ap_predict(x):
    ap = AP(x)
    cluster_centers_indices, labels = ap.predict()
    return cluster_centers_indices, labels

