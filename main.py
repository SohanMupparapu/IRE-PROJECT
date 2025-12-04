import json
import requests
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional
import pickle
from datetime import datetime
from elasticsearch import Elasticsearch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix
from scipy import stats
import hashlib


class NewsRankingSystem:
    """Personalized news ranking with LTR, XGBoost, Collaborative Filtering, and A/B testing"""
    
    def __init__(self, api_url="http://localhost:3000", es_host="localhost:9200"):
        self.api_url = api_url
        self.es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "G7r40vJX")) if es_host else None
        self.logs = []
        self.user_histories = defaultdict(list)
        self.articles = []
        
        # Models
        self.ltr_weights = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        
        # Collaborative filtering
        self.user_item_matrix = None
        self.svd_model = None
        self.cf_n_components = 50
        self.user_id_map = {}
        self.article_id_map = {}
        self.reverse_article_map = {}
        
        # Experiment tracking
        self.experiment_data = []
        self.variant_metrics = defaultdict(lambda: {
            'weighted_clicks': [],
            'users': set(),
            'queries': 0
        })
        
        # A/B test phase control
        self.ab_test_phase = "baseline"  # "baseline" or "experiment"
    
    def load_articles(self, filepath='articles.jsonl'):
        """Load articles from JSONL file (uuid, text, topics)"""
        self.articles = []
        with open(filepath, 'r') as f:
            for line in f:
                article = json.loads(line)
                self.articles.append({
                    'article_id': article['uuid'],
                    'text': article['text'],
                    'topics': article['topics']
                })
        print(f"Loaded {len(self.articles)} articles")
        return self.articles
    
    def index_articles_to_es(self):
        """Index articles to Elasticsearch"""
        if not self.es:
            print("Elasticsearch not configured")
            return
        
        index_name = 'news_articles'
        
        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
        
        mapping = {
            "mappings": {
                "properties": {
                    "article_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "topics": {"type": "keyword"}
                }
            }
        }
        self.es.indices.create(index=index_name, body=mapping)
        
        for article in self.articles:
            self.es.index(index=index_name, id=article['article_id'], document=article)
        
        print(f"Indexed {len(self.articles)} articles to Elasticsearch")
    
    def get_query(self):
        """Get query from simulation platform"""
        response = requests.get(f"{self.api_url}/query")
        return response.json()
    
    def submit_ranklist(self, query_id, user_id, ranked_article_ids):
        """Submit ranked list and get user actions"""
        payload = {
            "query_id": query_id,
            "user_id": user_id,
            "ranked_article_ids": ranked_article_ids
        }
        response = requests.post(f"{self.api_url}/ranklist", json=payload)
        return response.json()
    
    def elasticsearch_rank(self, query_text, top_k=20):
        """Baseline ranking using Elasticsearch BM25"""
        if self.es:
            results = self.es.search(
                index='news_articles',
                body={
                    "query": {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["text^2", "topics"]
                        }
                    },
                    "size": top_k
                }
            )
            return [hit['_id'] for hit in results['hits']['hits']]
        else:
            # Fallback: simple text matching
            scored_articles = []
            query_terms = query_text.lower().split()
            
            for article in self.articles:
                text = f"{article['text']} {' '.join(article.get('topics', []))}".lower()
                score = sum(text.count(term) for term in query_terms)
                scored_articles.append((article['article_id'], score))
            
            scored_articles.sort(key=lambda x: x[1], reverse=True)
            return [aid for aid, _ in scored_articles[:top_k]]
    
    def calculate_propensity(self, position):
        """Calculate position-based propensity score for IPS"""
        return 1.0 / np.log2(position + 2)
    
    def calculate_reward(self, actions):
        """Calculate reward from user actions with position-aware weighting"""
        if not actions:
            return 0.0
        
        reward = 0.0
        for action in actions:
            if action == 'Click':
                reward += 1.0
            elif action == 'Like':
                reward += 3.0
            elif action == 'Share':
                reward += 5.0
            elif action == 'Bookmark':
                reward += 4.0
            elif isinstance(action, dict) and 'Dwell' in action:
                dwell_secs = action['Dwell'].get('secs', 0)
                reward += min(dwell_secs / 10.0, 2.0)
        
        return reward
    
    def calculate_weighted_clicks(self, actions, top_k=10):
        """Calculate position-weighted clicks metric (primary evaluation metric)"""
        weighted_score = 0.0
        
        for position, action_list in enumerate(actions[:top_k]):
            if not action_list:
                continue
            
            # Position weight: higher positions get more weight
            position_weight = 1.0 / np.log2(position + 2)
            
            # Action rewards
            reward = self.calculate_reward(action_list)
            
            weighted_score += reward * position_weight
        
        return weighted_score
    
    def extract_features(self, article_id, query_text, user_id, position):
        """Extract features for ranking models"""
        query_terms = set(query_text.lower().split())
        article_obj = next((a for a in self.articles if a['article_id'] == article_id), None)
        
        if not article_obj:
            return np.zeros(15)
        
        text = article_obj.get('text', '').lower()
        topics = article_obj.get('topics', [])
        
        # Text relevance features
        text_match = sum(1 for term in query_terms if term in text) / max(len(query_terms), 1)
        topic_text = ' '.join(topics).lower()
        topic_match_query = sum(1 for term in query_terms if term in topic_text) / max(len(query_terms), 1)
        
        # User preference features
        user_history = self.user_histories.get(user_id, [])
        user_topic_prefs = defaultdict(float)
        
        for log in user_history[-50:]:  # Last 50 interactions
            if log.get('reward', 0) > 0:
                for topic in log.get('topics', []):
                    user_topic_prefs[topic] += log['reward']
        
        topic_preference_score = 0.0
        if user_topic_prefs:
            topic_preference_score = sum(user_topic_prefs.get(topic, 0) for topic in topics)
            topic_preference_score /= sum(user_topic_prefs.values())
        
        # Position features
        position_bias = 1.0 / np.log2(position + 2)
        
        # Content features
        num_topics = len(topics)
        text_length = len(text) / 1000.0
        
        # Historical performance
        article_clicks = sum(1 for log in self.logs 
                           if log.get('article_id') == article_id and log.get('clicked', False))
        article_views = sum(1 for log in self.logs if log.get('article_id') == article_id)
        article_ctr = article_clicks / max(article_views, 1)
        
        avg_reward = np.mean([log.get('reward', 0) for log in self.logs 
                             if log.get('article_id') == article_id]) if article_views > 0 else 0
        
        # User activity
        user_activity = len(user_history) / 100.0
        user_avg_reward = np.mean([log.get('reward', 0) for log in user_history]) if user_history else 0
        
        # Collaborative filtering score (if available)
        cf_score = self.get_cf_score(user_id, article_id)
        
        features = np.array([
            text_match,
            topic_match_query,
            topic_preference_score,
            position_bias,
            num_topics,
            text_length,
            article_ctr,
            avg_reward,
            user_activity,
            user_avg_reward,
            cf_score,
            np.log1p(article_views),
            np.log1p(article_clicks),
            len(user_history) > 0,  # Has history
            1.0  # Bias term
        ])
        
        return features
    
    def get_cf_score(self, user_id, article_id):
        """Get collaborative filtering score"""
        if self.svd_model is None or user_id not in self.user_id_map or article_id not in self.article_id_map:
            return 0.0
        
        try:
            user_idx = self.user_id_map[user_id]
            article_idx = self.article_id_map[article_id]
            
            # Get user and item latent factors
            user_factors = self.svd_model.components_[:, user_idx]
            item_factors = self.svd_model.components_[:, article_idx]
            
            score = np.dot(user_factors, item_factors)
            return float(score)
        except:
            return 0.0
    
    def train_collaborative_filtering(self):
        """Train collaborative filtering model using SVD"""
        print("Training collaborative filtering model...")
        
        if len(self.logs) < 20:
            print("Not enough data for CF")
            return
        
        # Build user-item interaction matrix
        users = list(set(log['user_id'] for log in self.logs))
        articles = list(set(log['article_id'] for log in self.logs))
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(users)}
        self.article_id_map = {aid: idx for idx, aid in enumerate(articles)}
        self.reverse_article_map = {idx: aid for aid, idx in self.article_id_map.items()}
        
        # Create sparse matrix
        rows, cols, data = [], [], []
        for log in self.logs:
            user_idx = self.user_id_map[log['user_id']]
            article_idx = self.article_id_map[log['article_id']]
            reward = log.get('reward', 0)
            
            rows.append(user_idx)
            cols.append(article_idx)
            data.append(reward)
        
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(users), len(articles))
        )
        
        # Apply SVD
        n_components = min(self.cf_n_components, min(self.user_item_matrix.shape) - 1)
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd_model.fit(self.user_item_matrix.T)  # Transpose for article-based
        
        print(f"CF model trained with {len(users)} users and {len(articles)} articles")
    
    def collaborative_filtering_rank(self, user_id, query_text, top_k=20):
        """Rank using collaborative filtering"""
        if self.svd_model is None or user_id not in self.user_id_map:
            return self.elasticsearch_rank(query_text, top_k)
        
        # Get candidate articles from baseline
        candidates = self.elasticsearch_rank(query_text, top_k * 2)
        
        # Score with CF
        scored_articles = []
        for article_id in candidates:
            cf_score = self.get_cf_score(user_id, article_id)
            scored_articles.append((article_id, cf_score))
        
        scored_articles.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scored_articles[:top_k]]
    
    def train_ltr_model(self, propensity_clip=0.1, learning_rate=0.01, iterations=100):
        """Train LTR model with IPS weighting"""
        print(f"Training LTR model with {len(self.logs)} samples...")
        
        if len(self.logs) < 10:
            print("Not enough data to train LTR")
            return
        
        # Initialize weights
        self.ltr_weights = np.random.randn(15) * 0.01
        
        for iteration in range(iterations):
            gradients = np.zeros(15)
            total_loss = 0
            count = 0
            
            for log in self.logs:
                features = log['features']
                reward = log['reward']
                position = log['position']
                
                # Calculate propensity score
                propensity = max(self.calculate_propensity(position), propensity_clip)
                
                # IPS weight
                ips_weight = reward / propensity
                
                # Prediction
                prediction = np.dot(features, self.ltr_weights)
                
                # Gradient descent with IPS weighting
                error = ips_weight - prediction
                gradients += error * features
                total_loss += error ** 2
                count += 1
            
            # Update weights
            if count > 0:
                self.ltr_weights += learning_rate * gradients / count
                
                if (iteration + 1) % 20 == 0:
                    avg_loss = total_loss / count
                    print(f"Iteration {iteration + 1}/{iterations}, Loss: {avg_loss:.4f}")
    
    def ltr_rank(self, query_text, user_id, top_k=20):
        """Rank using trained LTR model"""
        if self.ltr_weights is None:
            return self.elasticsearch_rank(query_text, top_k)
        
        scored_articles = []
        
        for i, article in enumerate(self.articles):
            features = self.extract_features(
                article['article_id'],
                query_text,
                user_id,
                position=i
            )
            score = np.dot(features, self.ltr_weights)
            scored_articles.append((article['article_id'], score))
        
        scored_articles.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scored_articles[:top_k]]
    
    def train_xgboost_model(self, propensity_clip=0.1):
        """Train XGBoost model with IPS weighting"""
        print(f"Training XGBoost model with {len(self.logs)} samples...")
        
        if len(self.logs) < 20:
            print("Not enough data to train XGBoost")
            return
        
        # Prepare training data
        X = np.array([log['features'] for log in self.logs])
        y = np.array([log['reward'] for log in self.logs])
        
        # Calculate IPS weights
        weights = []
        for log in self.logs:
            propensity = max(self.calculate_propensity(log['position']), propensity_clip)
            weights.append(1.0 / propensity)
        weights = np.array(weights)
        
        # Train XGBoost
        dtrain = xgb.DMatrix(X, label=y, weight=weights)
        
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        self.xgb_model = xgb.train(params, dtrain, num_boost_round=100)
        print("XGBoost model trained")
    
    def xgboost_rank(self, query_text, user_id, top_k=20):
        """Rank using XGBoost model"""
        if self.xgb_model is None:
            return self.elasticsearch_rank(query_text, top_k)
        
        scored_articles = []
        
        for i, article in enumerate(self.articles):
            features = self.extract_features(
                article['article_id'],
                query_text,
                user_id,
                position=i
            )
            features_matrix = xgb.DMatrix(features.reshape(1, -1))
            score = self.xgb_model.predict(features_matrix)[0]
            scored_articles.append((article['article_id'], score))
        
        scored_articles.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scored_articles[:top_k]]
    
    def get_variant_for_user(self, user_id):
        """Deterministic hash-based variant assignment"""
        if self.ab_test_phase == "baseline":
            return "baseline"
        
        # Use MD5 hash for consistent assignment across queries
        hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest()[:8], 16)
        
        # Map to variant (25% each)
        variant_idx = hash_val % 4
        variants = ["baseline", "ltr", "xgboost", "collaborative"]
        return variants[variant_idx]
    
    def get_ranking_by_variant(self, variant, query_text, user_id, top_k=20):
        """Get ranking based on assigned variant"""
        if variant == "ltr" and self.ltr_weights is not None:
            return self.ltr_rank(query_text, user_id, top_k), "ltr"
        elif variant == "xgboost" and self.xgb_model is not None:
            return self.xgboost_rank(query_text, user_id, top_k), "xgboost"
        elif variant == "collaborative" and self.svd_model is not None:
            return self.collaborative_filtering_rank(user_id, query_text, top_k), "collaborative"
        else:
            return self.elasticsearch_rank(query_text, top_k), "baseline"
    
    def collect_data_with_ab_testing(self, num_queries=100):
        """Collect interaction data using A/B testing"""
        print(f"Collecting {num_queries} queries (phase: {self.ab_test_phase})...")
        
        # Track variant assignments
        variant_counts = defaultdict(int)
        
        for i in range(num_queries):
            try:
                # Get query
                query_data = self.get_query()
                query_id = query_data['query_id']
                user_id = query_data['user_id']
                query_text = query_data['query_text']
                
                # Get variant assignment
                variant = self.get_variant_for_user(user_id)
                variant_counts[variant] += 1
                
                # Get ranked list based on variant
                ranked_ids, ranking_method = self.get_ranking_by_variant(
                    variant, query_text, user_id
                )
                
                # Submit and get actions
                result = self.submit_ranklist(query_id, user_id, ranked_ids)
                actions = result['actions']
                
                # Calculate weighted clicks metric
                weighted_clicks = self.calculate_weighted_clicks(actions)
                
                # Track metrics by variant
                self.variant_metrics[variant]['weighted_clicks'].append(weighted_clicks)
                self.variant_metrics[variant]['users'].add(user_id)
                self.variant_metrics[variant]['queries'] += 1
                
                # Log data for training
                for position, (article_id, action_list) in enumerate(zip(ranked_ids, actions)):
                    features = self.extract_features(article_id, query_text, user_id, position)
                    reward = self.calculate_reward(action_list)
                    
                    article_obj = next((a for a in self.articles if a['article_id'] == article_id), None)
                    topics = article_obj.get('topics', []) if article_obj else []
                    
                    log_entry = {
                        'query_id': query_id,
                        'user_id': user_id,
                        'query_text': query_text,
                        'article_id': article_id,
                        'position': position,
                        'features': features,
                        'actions': action_list,
                        'reward': reward,
                        'clicked': 'Click' in action_list if action_list else False,
                        'topics': topics,
                        'timestamp': datetime.now().isoformat(),
                        'variant': variant,
                        'ranking_method': ranking_method
                    }
                    
                    self.logs.append(log_entry)
                    self.user_histories[user_id].append(log_entry)
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{num_queries} queries")
                    print("Variant distribution:")
                    for v in sorted(variant_counts.keys()):
                        pct = (variant_counts[v] / (i+1)) * 100
                        print(f"  {v}: {variant_counts[v]} ({pct:.1f}%)")
                    
            except Exception as e:
                print(f"Error processing query {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nFinal variant distribution:")
        total = sum(variant_counts.values())
        for v in sorted(variant_counts.keys()):
            pct = (variant_counts[v] / total) * 100 if total > 0 else 0
            print(f"  {v}: {variant_counts[v]} ({pct:.1f}%)")
        
        print(f"Collected {len(self.logs)} total interaction logs")
    
    def run_experiment(self, initial_queries=100, experiment_queries=200):
        """Run complete experiment with A/B testing"""
        print("\n=== News Ranking A/B Testing Experiment ===")
        
        # Phase 1: Collect initial baseline data
        print("\n--- Phase 1: Initial Data Collection (Baseline Only) ---")
        self.ab_test_phase = "baseline"
        self.collect_data_with_ab_testing(num_queries=initial_queries)
        
        # Phase 2: Train all models
        print("\n--- Phase 2: Training Models ---")
        self.train_collaborative_filtering()
        self.train_ltr_model(iterations=100, learning_rate=0.01, propensity_clip=0.1)
        self.train_xgboost_model(propensity_clip=0.1)
        
        # Phase 3: Run A/B test
        print("\n--- Phase 3: A/B Testing ---")
        self.ab_test_phase = "experiment"
        
        # Reset variant metrics for clean experiment
        self.variant_metrics = defaultdict(lambda: {
            'weighted_clicks': [],
            'users': set(),
            'queries': 0
        })
        
        self.collect_data_with_ab_testing(num_queries=experiment_queries)
        
        # Phase 4: Analyze results
        print("\n--- Phase 4: Results Analysis ---")
        results = self.analyze_results()
        
        return results
    
    def analyze_results(self):
        """Analyze A/B test results with statistical significance"""
        print("\n=== A/B Test Results: Weighted Clicks Metric ===")
        
        results = {}
        
        # Get baseline data
        baseline_values = self.variant_metrics['baseline']['weighted_clicks']
        
        if not baseline_values:
            print("No baseline data available")
            return results
        
        baseline_mean = np.mean(baseline_values)
        baseline_std = np.std(baseline_values)
        n_baseline = len(baseline_values)
        
        print(f"\n{'='*70}")
        print(f"BASELINE")
        print(f"{'='*70}")
        print(f"Mean Weighted Clicks: {baseline_mean:.4f} (±{baseline_std:.4f})")
        print(f"Sample size: {n_baseline}")
        print(f"Unique users: {len(self.variant_metrics['baseline']['users'])}")
        
        # Analyze each variant
        for variant in ['ltr', 'xgboost', 'collaborative']:
            variant_values = self.variant_metrics[variant]['weighted_clicks']
            
            if not variant_values:
                print(f"\n{'='*70}")
                print(f"{variant.upper()}")
                print(f"{'='*70}")
                print("No data collected for this variant")
                continue
            
            variant_mean = np.mean(variant_values)
            variant_std = np.std(variant_values)
            n_variant = len(variant_values)
            
            # Calculate improvement
            improvement = ((variant_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
            
            # Welch's t-test
            s1_sq, s2_sq = baseline_std**2, variant_std**2
            se = np.sqrt((s1_sq / n_baseline) + (s2_sq / n_variant))
            t_stat = (variant_mean - baseline_mean) / se if se > 0 else 0
            
            # Degrees of freedom
            if s1_sq > 0 and s2_sq > 0:
                df_num = ((s1_sq / n_baseline) + (s2_sq / n_variant))**2
                df_denom = ((s1_sq / n_baseline)**2 / (n_baseline - 1)) + ((s2_sq / n_variant)**2 / (n_variant - 1))
                df = df_num / df_denom if df_denom > 0 else 1
            else:
                df = min(n_baseline, n_variant) - 1
            
            # p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            # Confidence interval
            ci_margin = 1.96 * se
            ci_lower = (variant_mean - baseline_mean) - ci_margin
            ci_upper = (variant_mean - baseline_mean) + ci_margin
            
            results[variant] = {
                'mean': variant_mean,
                'std': variant_std,
                'baseline_mean': baseline_mean,
                'improvement': improvement,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_95': p_value < 0.05,
                'significant_99': p_value < 0.01,
                'confidence_interval': (ci_lower, ci_upper),
                'sample_size': n_variant
            }
            
            print(f"\n{'='*70}")
            print(f"{variant.upper()}")
            print(f"{'='*70}")
            print(f"Mean Weighted Clicks: {variant_mean:.4f} (±{variant_std:.4f})")
            print(f"Sample size: {n_variant}")
            print(f"Unique users: {len(self.variant_metrics[variant]['users'])}")
            print(f"\nStatistical Analysis:")
            print(f"  Improvement: {improvement:+.2f}%")
            print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Significant (95%): {'✓ Yes' if results[variant]['significant_95'] else '✗ No'}")
            print(f"  Significant (99%): {'✓ Yes' if results[variant]['significant_99'] else '✗ No'}")
        
        # Determine winner
        print(f"\n{'='*70}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*70}")
        
        if results:
            best_variant = max(
                [(v, r['mean']) for v, r in results.items()],
                key=lambda x: x[1],
                default=('baseline', baseline_mean)
            )
            
            print(f"Best performing variant: {best_variant[0].upper()}")
            print(f"Best mean weighted clicks: {best_variant[1]:.4f}")
            
            significant_winners = [v for v, r in results.items() 
                                  if r['significant_95'] and r['improvement'] > 0]
            
            if significant_winners:
                print(f"\n✓ Significant improvements found: {', '.join(significant_winners)}")
                print("Recommendation: Deploy best performing variant")
            else:
                print("\n✗ No significant improvements found")
                print("Recommendation: Continue with baseline or collect more data")
        else:
            print("No variant data to compare")
        
        return results
    
    def save_model(self, filepath='ranking_model.pkl'):
        """Save trained models"""
        model_data = {
            'ltr_weights': self.ltr_weights,
            'xgb_model': self.xgb_model,
            'svd_model': self.svd_model,
            'user_id_map': self.user_id_map,
            'article_id_map': self.article_id_map,
            'logs': self.logs,
            'user_histories': dict(self.user_histories),
            'variant_metrics': {k: {
                'weighted_clicks': v['weighted_clicks'],
                'users': list(v['users']),
                'queries': v['queries']
            } for k, v in self.variant_metrics.items()}
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Models saved to {filepath}")
    
    def load_model(self, filepath='ranking_model.pkl'):
        """Load trained models"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.ltr_weights = model_data['ltr_weights']
        self.xgb_model = model_data.get('xgb_model')
        self.svd_model = model_data.get('svd_model')
        self.user_id_map = model_data.get('user_id_map', {})
        self.article_id_map = model_data.get('article_id_map', {})
        self.logs = model_data['logs']
        self.user_histories = defaultdict(list, model_data['user_histories'])
        print(f"Models loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize system
    system = NewsRankingSystem(api_url="http://localhost:3000")
    
    # Load articles (uuid, text, topics format)
    system.load_articles('./data/articles.jsonl')
    
    # Optional: Index to Elasticsearch
    system.index_articles_to_es()
    
    # Run complete experiment
    print("\n🚀 Starting News Ranking A/B Test Experiment")
    results = system.run_experiment(
        initial_queries=100,  # Initial baseline data collection
        experiment_queries=300  # A/B test queries across all variants
    )
    
    # Save models
    system.save_model('news_ranking_models.pkl')
    
    print("\n✅ Experiment Complete!")
    print(f"Total logs collected: {len(system.logs)}")
    print(f"Unique users: {len(system.user_histories)}")
    print("\nModels trained:")
    print(f"  - LTR: {'✓' if system.ltr_weights is not None else '✗'}")
    print(f"  - XGBoost: {'✓' if system.xgb_model is not None else '✗'}")
    print(f"  - Collaborative Filtering: {'✓' if system.svd_model is not None else '✗'}")