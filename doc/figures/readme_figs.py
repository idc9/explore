from sklearn.datasets import make_classification
# from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd

from explore.SingleBlock import SingleBlock

X, y = make_classification(n_samples=400,
                           n_features=5,
                           n_informative=3,
                           n_redundant=0,
                           n_classes=3,
                           n_clusters_per_class=1,
                           random_state=0)

# fit PCA
# pca = PCA(n_components=3).fit_transform(X)

# # create pd.DataFrame so we can have variable names
# df = pd.DataFrame(pca,
#                   columns=['pca_comp_{}'.format(i + 1) for i in range(pca.shape[1])])
# df['classes'] = y
# df['classes'] = df['class'].astype('category')

# explore is happiest when you pass it pandas objects
df = pd.DataFrame(X,
                  columns=['feat_{}'.format(i + 1) for i in range(X.shape[1])])
df['classes'] = y
# tell explore that 'classes' is a categorical variable
df['classes'] = df['classes'].astype('category')
# df['classes'] = df['classes'].astype(str)

comparisons = SingleBlock(multi_test='fdr_bh',
                          cat_test='auc', multi_cat='ovo')
comparisons.fit(df)
# benjamini hochberg for multiple testing correction
comparisons.correct_multi_tests()
comparisons.plot()
plt.savefig('comparisons.png')
