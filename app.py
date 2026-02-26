from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

class FunkSVD:
    def __init__(self, n_users, n_items, n_factors=20, lr=0.005, reg=0.02, n_epochs=50):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.mu = 0
        self.P = np.random.normal(0, 0.1, (n_users, n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)

    def predict(self, user_id, item_id):
        u, v = user_id - 1, item_id - 1
        pred = self.mu + self.bu[u] + self.bi[v] + self.P[u] @ self.Q[v]
        return float(np.clip(pred, 1, 5))

# 加载模型和数据
with open('funksvd_model.pkl', 'rb') as f:
    model = pickle.load(f)

train_data = pd.read_csv('train_data.csv')

genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies_full = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                          names=['item_id', 'title', 'release_date', 'video_date', 'imdb_url',
                                 'unknown'] + genre_cols)

genre_matrix = movies_full[genre_cols].values
item_similarity = cosine_similarity(genre_matrix)

print('模型加载完成')


def hybrid_recommend(user_id, alpha=0.7, top_n=10):
    user_rated = train_data[train_data.user_id == user_id]
    if len(user_rated) == 0:
        return None, '用户不存在'

    rated_items = user_rated.item_id.values
    all_items = movies_full.item_id.values
    unrated_items = np.setdiff1d(all_items, rated_items)
    liked_items = user_rated[user_rated.rating >= 4].item_id.values

    results = []
    for item_id in unrated_items:
        cf_score = (model.predict(user_id, item_id) - 1) / 4
        item_idx = item_id - 1
        if len(liked_items) > 0:
            liked_idx = liked_items - 1
            liked_idx = liked_idx[liked_idx < item_similarity.shape[0]]
            content_score = item_similarity[item_idx][liked_idx].mean() if len(liked_idx) > 0 else 0.5
        else:
            content_score = 0.5
        hybrid_score = alpha * cf_score + (1 - alpha) * content_score
        results.append((item_id, cf_score, content_score, hybrid_score))

    results.sort(key=lambda x: x[3], reverse=True)

    output = []
    for item_id, cf_score, content_score, hybrid_score in results[:top_n]:
        movie = movies_full[movies_full.item_id == item_id].iloc[0]
        genres = [g for g in genre_cols if movie[g] == 1]
        output.append({
            'rank': len(output) + 1,
            'title': movie['title'],
            'genres': ', '.join(genres),
            'cf_score': round(cf_score * 4 + 1, 2),
            'hybrid_score': round(hybrid_score, 4)
        })
    return output, None


def get_user_preference(user_id):
    liked = train_data[(train_data.user_id == user_id) & (train_data.rating >= 4)]
    liked_movies = movies_full[movies_full.item_id.isin(liked.item_id)]
    pref = liked_movies[genre_cols].sum().sort_values(ascending=False)
    return [{'genre': g, 'count': int(c)} for g, c in pref.items() if c > 0]


def get_user_history(user_id, top_n=10):
    user_ratings = train_data[train_data.user_id == user_id].copy()
    user_ratings = user_ratings.sort_values('rating', ascending=False).head(top_n)
    result = []
    for row in user_ratings.itertuples():
        title = movies_full[movies_full.item_id == row.item_id]['title'].values
        if len(title) > 0:
            result.append({'title': title[0], 'rating': row.rating})
    return result


@app.route('/')
def index():
    max_user = int(train_data.user_id.max())
    return render_template('index.html', max_user=max_user)


@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        user_id = int(request.args.get('user_id', 1))
        top_n = int(request.args.get('top_n', 10))
        alpha = float(request.args.get('alpha', 0.7))
    except ValueError:
        return jsonify({'error': '参数格式错误'}), 400

    recommendations, error = hybrid_recommend(user_id, alpha=alpha, top_n=top_n)
    if error:
        return jsonify({'error': error}), 404

    return jsonify({
        'user_id': user_id,
        'recommendations': recommendations,
        'history': get_user_history(user_id),
        'preferences': get_user_preference(user_id)
    })


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
