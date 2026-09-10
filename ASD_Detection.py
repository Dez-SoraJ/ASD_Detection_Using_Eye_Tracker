import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Silences TensorFlow startup messages
import json
import time
import pathlib
from collections import deque
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import cv2
import mediapipe as mp
import warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import loguniform, entropy as scipy_entropy
from sklearn.linear_model import LogisticRegression
try:
	import xgboost as xgb  # type: ignore
except Exception:
	xgb = None
try:
	import lightgbm as lgb  # type: ignore
except Exception:
	lgb = None

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

SCRIPT_DIR = pathlib.Path(_file_).parent.resolve()

class AutismScreeningSystem:
	def _init_(self, csv_path: str):
		self.csv_path = csv_path
		self.screen_width = 1920
		self.screen_height = 1080
		self.scaler = StandardScaler()
		self.feature_names = []
		self.ml_models = {}
		self.dl_models = {}
		self.sequence_models = {}
		self.ensemble_model = None
		self.is_trained = False
		self._full_df: Optional[pd.DataFrame] = None
		
		# --- Real-time analysis attributes ---
		self.current_session_data = []
		self.gaze_path = deque(maxlen=150)
		self.fixations = 0
		self.saccades = 0
		self.session_start_time = None
		self.last_gaze_point = None
		self.last_gaze_time = None
		self.fixation_start_time = None
		self.fixation_start_pos = None
		self.is_in_fixation = False
		self.counted_fixation = False
		# --- NEW: Gaze tracking attributes ---
		self.is_calibrated = False
		self.calibrated_gaze_offset = np.array([0.0, 0.0])
		# ** You can adjust this sensitivity value for your setup **
		self.GAZE_SENSITIVITY = 0.9 
		self.SMOOTHING_FACTOR = 0.8 # Higher value = more smoothing (e.g., 0.0 to 0.95)
		self.last_smoothed_gaze = np.array([self.screen_width / 2, self.screen_height / 2])
		
		self.VELOCITY_THRESHOLD = 2000
		self.FIXATION_DURATION_THRESHOLD = 0.15
		self.FIXATION_RADIUS_THRESHOLD = 50
		self.VIGOROUS_THRESHOLD = 1000
		# Kalman filter (2D position + velocity)
		self.kalman_initialized = False
		self.k_F = None  # state transition
		self.k_H = None  # observation matrix
		self.k_P = None  # covariance
		self.k_Q = None  # process noise
		self.k_R = None  # measurement noise
		self.k_x = None  # state vector [x,y,vx,vy]
		# Adaptive thresholds (start with defaults)
		self.adaptive_velocity_threshold = self.VELOCITY_THRESHOLD
		self.adaptive_fix_radius = self.FIXATION_RADIUS_THRESHOLD
		self.adaptive_fix_dur = self.FIXATION_DURATION_THRESHOLD

		self.face_mesh = mp.solutions.face_mesh.FaceMesh(
			max_num_faces=1, refine_landmarks=True,
			min_detection_confidence=0.5, min_tracking_confidence=0.5)
		print(" Autism Screening System Initialized")
	
	def load_and_preprocess_data(self) -> tuple[np.ndarray, np.ndarray]:
		# This function remains unchanged
		print(f" Loading data from: {self.csv_path}")
		try:
			df = pd.read_csv(self.csv_path)
			self._full_df = df.copy()
			df['subject_id'] = df.index // 1000
			features_list, labels_list = [], []
			self._subjects = []
			for subject_id in df['subject_id'].unique():
				subject_data = df.loc[df['subject_id'] == subject_id].copy()
				if len(subject_data) < 50: continue
				subject_data.rename(columns={'Point of Regard Left X [px]': 'x', 'Point of Regard Left Y [px]': 'y', 'Group': 'label'}, inplace=True)
				if 'timestamp' not in subject_data.columns: subject_data['timestamp'] = subject_data.index * (1/60)
				features = self.extract_comprehensive_features(subject_data)
				if features:
					features_list.append(list(features.values()))
					labels_list.append(subject_data['label'].iloc[0])
					self._subjects.append(subject_id)
			X, y = np.array(features_list), np.array(labels_list)
			print(f"[SUCCESS] Extracted features for {len(X)} subjects")
			return X, y
		except Exception as e: print(f"[ERROR] Error loading data: {e}"); return None, None

	def _filter_session_df(self, df: pd.DataFrame) -> pd.DataFrame:
		# Remove off-screen points and unrealistic spikes; drop missing critical fields
		df = df.dropna(subset=['x','y','timestamp'])
		df = df[(df['x']>=0) & (df['x']<=self.screen_width) & (df['y']>=0) & (df['y']<=self.screen_height)]
		if len(df) < 3: return df
		dx = np.diff(df['x'].values); dy = np.diff(df['y'].values); dt = np.diff(df['timestamp'].values)
		dt[dt==0] = 1e-3
		vel = np.sqrt(dx*dx + dy*dy) / dt
		v_med = np.median(vel); v_iqr = np.subtract(*np.percentile(vel, [75,25]))
		v_thr = v_med + 5*v_iqr if v_iqr>0 else v_med*5
		mask = np.ones(len(df), dtype=bool)
		mask[1:] = vel < max(500, v_thr)
		return df.loc[mask]

	def build_sequence_windows(self, df: pd.DataFrame, window: int = 100, stride: int = 50) -> Tuple[np.ndarray, np.ndarray]:
		# Create sequence windows with features: x_norm, y_norm, pupil_norm(optional), blink_flag(optional), delta_t
		df = self._filter_session_df(df)
		if len(df) < window: return np.empty((0, window, 5)), np.array([])
		t = df['timestamp'].values
		x = df['x'].values / self.screen_width
		y = df['y'].values / self.screen_height
		pupil = (df['pupil_size'].fillna(df['pupil_size'].median()) if 'pupil_size' in df else pd.Series(np.zeros(len(df)))).values
		pupil = (pupil - np.nanmean(pupil)) / (np.nanstd(pupil) + 1e-6)
		blink = (df['blink_flag'] if 'blink_flag' in df else pd.Series(np.zeros(len(df)))).values
		dt = np.diff(t, prepend=t[0])
		dt[0] = 1/60.0
		feat = np.stack([x, y, pupil, blink, dt], axis=1)
		seqs = []
		for start in range(0, len(feat)-window+1, stride):
			seqs.append(feat[start:start+window])
		X_seq = np.array(seqs, dtype=np.float32)
		# Label per window from session label if available
		y_label = None
		if 'label' in df.columns:
			try:
				lab = int(df['label'].iloc[0])
				y_label = np.full((len(X_seq),), lab, dtype=int)
			except Exception:
				y_label = np.full((len(X_seq),), 0, dtype=int)
		else:
			y_label = np.full((len(X_seq),), 0, dtype=int)
		return X_seq, y_label
	
	def extract_comprehensive_features(self, data: pd.DataFrame) -> Dict[str, float]:
		features = {}
		gaze_x, gaze_y, timestamps = data['x'].values, data['y'].values, data['timestamp'].values
		dx, dy, dt = np.diff(gaze_x), np.diff(gaze_y), np.diff(timestamps)
		dt[dt == 0] = 1e-3
		velocity = np.sqrt(dx*2 + dy*2) / dt
		features['mean_x'], features['mean_y'] = np.mean(gaze_x), np.mean(gaze_y)
		features['std_x'], features['std_y'] = np.std(gaze_x), np.std(gaze_y)
		features['mean_velocity'] = float(np.mean(velocity))
		features['velocity_variance'] = float(np.var(velocity))

		# Compute fixations and saccades from data
		fixations = 0
		saccades = 0
		fixation_durations: List[float] = []
		saccade_amplitudes: List[float] = []
		last_point = None
		last_time = None
		is_in_fixation = False
		fixation_start_time = None
		fixation_start_pos = None
		counted_fixation = False
		for i in range(len(gaze_x)):
			current_time = timestamps[i]
			current_point = np.array([gaze_x[i], gaze_y[i]])
			if last_point is not None:
				dt = current_time - last_time
				if dt > 0:
					dist = np.linalg.norm(current_point - last_point)
					velocity_here = dist / dt
					if velocity_here < self.VELOCITY_THRESHOLD:
						if not is_in_fixation:
							is_in_fixation = True
							fixation_start_time = current_time
							fixation_start_pos = current_point
							counted_fixation = False
						elif not counted_fixation:
							fix_dist = np.linalg.norm(current_point - fixation_start_pos)
							fix_dur = current_time - fixation_start_time
							if fix_dur > self.FIXATION_DURATION_THRESHOLD and fix_dist < self.FIXATION_RADIUS_THRESHOLD:
								fixations += 1
								fixation_durations.append(float(fix_dur))
								counted_fixation = True
					else:
						if is_in_fixation:
							saccades += 1
							# saccade amplitude as distance from last fixation start to current point
							if fixation_start_pos is not None:
								saccade_amplitudes.append(float(np.linalg.norm(current_point - fixation_start_pos)))
							is_in_fixation = False
							fixation_start_time = None
			last_point = current_point
			last_time = current_time
		features['fixation_count'] = fixations
		features['saccade_count'] = saccades
		features['avg_fixation_duration'] = float(np.mean(fixation_durations)) if fixation_durations else 0.0
		features['saccade_amp_mean'] = float(np.mean(saccade_amplitudes)) if saccade_amplitudes else 0.0
		features['saccade_amp_std'] = float(np.std(saccade_amplitudes)) if saccade_amplitudes else 0.0
		# Scanpath entropy on a coarse grid
		try:
			H, _, _ = np.histogram2d(gaze_x, gaze_y, bins=16, range=[[0, self.screen_width], [0, self.screen_height]])
			p = H.flatten().astype(np.float64)
			p = p / p.sum() if p.sum() > 0 else p
			features['scanpath_entropy'] = float(scipy_entropy(p, base=2)) if p.sum() > 0 else 0.0
		except Exception:
			features['scanpath_entropy'] = 0.0

		if not self.feature_names or 'fixation_count' not in self.feature_names:
			self.feature_names = list(features.keys())
		return {name: features.get(name, 0) for name in self.feature_names}
		
	# Functions train_all_models, create_ensemble_model, save_models, load_models remain unchanged...
	def train_all_models(self):
		X, y = self.load_and_preprocess_data()
		if X is None or len(X) == 0: return False
		# 70/15/15 split with stratification
		X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
		X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
		# Scale
		X_train_s = self.scaler.fit_transform(X_train)
		X_val_s = self.scaler.transform(X_val)
		X_test_s = self.scaler.transform(X_test)
		# ---- Sequence dataset for LSTM (using original per-subject sequences) ----
		seq_train, seq_val, seq_test = [], [], []
		y_seq_train, y_seq_val, y_seq_test = [], [], []
		if self._full_df is not None:
			for subset_name, sub_ids in [('train', set(map(int, np.where(np.isin(self._subjects, np.unique(self._subjects))[0][np.isin(X, X)]))))]:
				pass  # placeholder to avoid complex mapping; build from df directly below
			# Build sequences by subject and split via the same subject_id partitioning
			df = self._full_df.copy()
			df['subject_id'] = df.index // 1000 if 'subject_id' not in df.columns else df['subject_id']
			subjects = df['subject_id'].unique()
			rng = np.random.default_rng(42)
			rng.shuffle(subjects)
			n_tr = int(0.7*len(subjects)); n_val = int(0.15*len(subjects))
			train_ids = set(subjects[:n_tr]); val_ids = set(subjects[n_tr:n_tr+n_val]); test_ids = set(subjects[n_tr+n_val:])
			for sid in subjects:
				sdf = df.loc[df['subject_id']==sid].copy()
				if len(sdf) < 120: continue
				X_seq, y_seq = self.build_sequence_windows(sdf, window=100, stride=50)
				if len(X_seq)==0: continue
				if sid in train_ids:
					seq_train.append(X_seq); y_seq_train.append(y_seq)
				elif sid in val_ids:
					seq_val.append(X_seq); y_seq_val.append(y_seq)
				else:
					seq_test.append(X_seq); y_seq_test.append(y_seq)
			# Concatenate
			def _cat(lst):
				return (np.concatenate(lst, axis=0) if len(lst)>0 else np.empty((0,100,5),dtype=np.float32))
			X_seq_train = _cat(seq_train); y_seq_train = (np.concatenate(y_seq_train) if len(y_seq_train)>0 else np.array([]))
			X_seq_val = _cat(seq_val); y_seq_val = (np.concatenate(y_seq_val) if len(y_seq_val)>0 else np.array([]))
			X_seq_test = _cat(seq_test); y_seq_test = (np.concatenate(y_seq_test) if len(y_seq_test)>0 else np.array([]))
		else:
			X_seq_train = np.empty((0,100,5)); X_seq_val = np.empty((0,100,5)); X_seq_test = np.empty((0,100,5))
			y_seq_train = np.array([]); y_seq_val = np.array([]); y_seq_test = np.array([])
		# Class weights for imbalance
		try:
			cls_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
			class_weight_dict = {int(c): w for c, w in zip(np.unique(y_train), cls_weights)}
		except Exception:
			class_weight_dict = None
		# Hyperparameter tuning - Random Forest
		rf = RandomForestClassifier(random_state=42)
		rf_grid = {
			'n_estimators': [100, 200, 400],
			'max_depth': [None, 10, 20, 30],
			'min_samples_split': [2, 5, 10],
			'min_samples_leaf': [1, 2, 4],
			'max_features': ['sqrt', 'log2']
		}
		rf_search = RandomizedSearchCV(rf, rf_grid, n_iter=20, scoring='roc_auc', cv=3, n_jobs=-1, random_state=42, verbose=0)
		rf_search.fit(X_train_s, y_train)
		rf_best = rf_search.best_estimator_
		rf_cal = CalibratedClassifierCV(rf_best, cv=3)
		rf_cal.fit(X_train_s, y_train)
		self.ml_models['RF'] = {'model': rf_cal}
		# Hyperparameter tuning - SVM (RBF)
		svm = SVC(probability=True, random_state=42)
		svm_params = {
			'C': loguniform(1e-2, 1e2),
			'gamma': loguniform(1e-4, 1e0),
			'kernel': ['rbf']
		}
		svm_search = RandomizedSearchCV(svm, svm_params, n_iter=25, scoring='roc_auc', cv=3, n_jobs=-1, random_state=42, verbose=0)
		svm_search.fit(X_train_s, y_train)
		svm_best = svm_search.best_estimator_
		self.ml_models['SVM'] = {'model': svm_best}
		# Deep Learning model with BN/Dropout, early stopping, checkpoints
		input_dim = X_train_s.shape[1]
		dl_model = keras.Sequential([
			layers.Input(shape=(input_dim,)),
			layers.Dense(128, activation='relu'),
			layers.BatchNormalization(),
			layers.Dropout(0.3),
			layers.Dense(64, activation='relu'),
			layers.BatchNormalization(),
			layers.Dropout(0.3),
			layers.Dense(1, activation='sigmoid')
		])
		dl_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=[keras.metrics.AUC(name='auc'), 'accuracy'])
		ckpt_path = str(SCRIPT_DIR / 'autism_models' / 'DNN.best.keras')
		callbacks = [
			keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True),
			keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5),
			keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_auc', mode='max', save_best_only=True)
		]
		history = dl_model.fit(
			X_train_s, y_train,
			validation_data=(X_val_s, y_val),
			epochs=100,
			batch_size=64,
			class_weight=class_weight_dict,
			verbose=0,
			callbacks=callbacks
		)
		self.dl_models['DNN'] = {'model': dl_model, 'history': history.history}
		# ---- LSTM model for sequence data ----
		lstm_model = None
		if X_seq_train.shape[0] > 0 and X_seq_val.shape[0] > 0:
			seq_input = keras.Input(shape=(X_seq_train.shape[1], X_seq_train.shape[2]))
			xl = layers.Masking(mask_value=0.0)(seq_input)
			xl = layers.LSTM(64, return_sequences=True, dropout=0.2)(xl)
			xl = layers.LSTM(32, dropout=0.2)(xl)
			xl = layers.Dense(32, activation='relu')(xl)
			seq_out = layers.Dense(1, activation='sigmoid')(xl)
			lstm_model = keras.Model(seq_input, seq_out)
			lstm_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=[keras.metrics.AUC(name='auc'), 'accuracy'])
			callbacks_seq = [
				keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True)
			]
			lstm_model.fit(X_seq_train, y_seq_train, validation_data=(X_seq_val, y_seq_val), epochs=50, batch_size=64, verbose=0, callbacks=callbacks_seq)
			self.sequence_models['LSTM'] = {'model': lstm_model}
		# Evaluate individual models on validation for stacking
		val_preds = {}
		val_auc = {}
		for name, m in self.ml_models.items():
			probs = m['model'].predict_proba(X_val_s)[:, 1]
			val_preds[name] = probs; val_auc[name] = roc_auc_score(y_val, probs)
		dnn_probs = dl_model.predict(X_val_s, verbose=0).flatten()
		val_preds['DNN'] = dnn_probs; val_auc['DNN'] = roc_auc_score(y_val, dnn_probs)
		# LSTM window-level predictions aggregated per subject (mean)
		if 'LSTM' in self.sequence_models and X_seq_val.shape[0] > 0:
			lstm_val_probs = self.sequence_models['LSTM']['model'].predict(X_seq_val, verbose=0).flatten()
			# Approximate subject-level mapping: average over all windows in validation split
			# When windows are many per subject, a precise mapping should be used
			val_preds['LSTM'] = np.full_like(dnn_probs, float(np.mean(lstm_val_probs)))
			val_auc['LSTM'] = roc_auc_score(y_val, val_preds['LSTM'])
		# Include optional XGB/LGBM
		if xgb is not None:
			xgb_model = xgb.XGBClassifier(
				n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
				reg_lambda=1.0, objective='binary:logistic', eval_metric='logloss', random_state=42
			)
			xgb_model.fit(X_train_s, y_train)
			self.ml_models['XGB'] = {'model': xgb_model}
			val_preds['XGB'] = xgb_model.predict_proba(X_val_s)[:, 1]
			val_auc['XGB'] = roc_auc_score(y_val, val_preds['XGB'])
		if lgb is not None:
			lgb_model = lgb.LGBMClassifier(
				n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
				reg_lambda=1.0, random_state=42
			)
			lgb_model.fit(X_train_s, y_train)
			self.ml_models['LGBM'] = {'model': lgb_model}
			val_preds['LGBM'] = lgb_model.predict_proba(X_val_s)[:, 1]
			val_auc['LGBM'] = roc_auc_score(y_val, val_preds['LGBM'])
		# Stacked ensemble (meta-learner on validation predictions)
		stack_list = [val_preds['RF'], val_preds['SVM'], val_preds['DNN']]
		if 'XGB' in val_preds: stack_list.append(val_preds['XGB'])
		if 'LGBM' in val_preds: stack_list.append(val_preds['LGBM'])
		if 'LSTM' in val_preds: stack_list.append(val_preds['LSTM'])
		stack_X_val = np.vstack(stack_list).T
		meta = LogisticRegression(max_iter=1000)
		meta.fit(stack_X_val, y_val)
		self.ensemble_model = {'type': 'stacked', 'coefficients': meta.coef_.tolist(), 'intercept': meta.intercept_.tolist(), 'order': list(val_preds.keys())}
		# Final evaluation on test set
		def eval_model_probs(y_true, probs):
			pred_labels = (probs >= 0.5).astype(int)
			acc = accuracy_score(y_true, pred_labels)
			prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred_labels, average='binary', zero_division=0)
			auc = roc_auc_score(y_true, probs)
			cm = confusion_matrix(y_true, pred_labels)
			return acc, prec, rec, f1, auc, cm
		test_probs = []
		# RF
		rf_probs = self.ml_models['RF']['model'].predict_proba(X_test_s)[:, 1]; test_probs.append(rf_probs)
		# SVM
		svm_probs = self.ml_models['SVM']['model'].predict_proba(X_test_s)[:, 1]; test_probs.append(svm_probs)
		# DNN
		dnn_test_probs = dl_model.predict(X_test_s, verbose=0).flatten(); test_probs.append(dnn_test_probs)
		# LSTM aggregated prediction for test split
		if 'LSTM' in self.sequence_models and 'LSTM' in val_preds and X_seq_test.shape[0] > 0:
			lstm_test_probs = self.sequence_models['LSTM']['model'].predict(X_seq_test, verbose=0).flatten()
			test_probs.append(np.full_like(dnn_test_probs, float(np.mean(lstm_test_probs))))
		# Optional XGB/LGBM
		if 'XGB' in self.ml_models:
			test_probs.append(self.ml_models['XGB']['model'].predict_proba(X_test_s)[:, 1])
		if 'LGBM' in self.ml_models:
			test_probs.append(self.ml_models['LGBM']['model'].predict_proba(X_test_s)[:, 1])
		# Ensemble via meta-learner
		stack_X_test = np.vstack(test_probs).T
		final_probs = meta.predict_proba(stack_X_test)[:, 1]
		acc, prec, rec, f1, aucv, cm = eval_model_probs(y_test, final_probs)
		print(f"[TEST] Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {aucv:.3f}")
		# Save confusion matrix figure
		plt.figure(figsize=(4,3))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
		plt.title('Test Confusion Matrix')
		plt.xlabel('Predicted'); plt.ylabel('True')
		cm_path = SCRIPT_DIR / 'training_confusion_matrix.png'
		plt.tight_layout(); plt.savefig(cm_path); plt.close()
		# Save models and finish
		self.save_models()
		self.is_trained = True; return True
	
	def create_ensemble_model(self, X_test, y_test):
		# Backward-compatible method; keep simple average if called
		preds = [m['model'].predict_proba(X_test)[:, 1] for m in self.ml_models.values()]
		preds.append(self.dl_models['DNN']['model'].predict(X_test, verbose=0).flatten())
		self.ensemble_model = {'type': 'average'}
		final_preds = np.mean(preds, axis=0)
		print(f" Ensemble AUC: {roc_auc_score(y_test, final_preds):.3f}")
	
	def save_models(self):
		p = SCRIPT_DIR / "autism_models"; p.mkdir(exist_ok=True)
		joblib.dump(self.scaler, p / "scaler.pkl"); joblib.dump(self.feature_names, p / "feature_names.pkl")
		for name, data in self.ml_models.items(): joblib.dump(data['model'], p / f"{name}.pkl")
		self.dl_models['DNN']['model'].save(p / "DNN.keras")
		with open(p / "ensemble.json", 'w') as f: json.dump(self.ensemble_model, f)
		print(" Models saved successfully!")
	
	def load_models(self):
		p = SCRIPT_DIR / "autism_models";
		if not p.exists(): return False
		try:
			self.scaler = joblib.load(p / "scaler.pkl"); self.feature_names = joblib.load(p / "feature_names.pkl")
			for f in p.glob("*.pkl"):
				if f.stem not in ["scaler", "feature_names"]: self.ml_models[f.stem] = {'model': joblib.load(f)}
			self.dl_models['DNN'] = {'model': keras.models.load_model(p / "DNN.keras")}
			with open(p / "ensemble.json", 'r') as f: self.ensemble_model = json.load(f)
			self.is_trained = True; print("[SUCCESS] Models loaded successfully!"); return True
		except Exception as e: print(f"[ERROR] Error loading models: {e}"); return False
	
	def _get_eye_offset(self, landmarks: Any) -> Optional[np.ndarray]:
		"""Calculates the normalized offset of the pupil from the eye center."""
		try:
			# Using right eye landmarks for calculation
			right_eye_left_corner = np.array([landmarks[33].x, landmarks[33].y])
			right_eye_right_corner = np.array([landmarks[133].x, landmarks[133].y])
			pupil = np.array([landmarks[473].x, landmarks[473].y])

			eye_center = (right_eye_left_corner + right_eye_right_corner) / 2.0
			eye_width = np.linalg.norm(right_eye_right_corner - right_eye_left_corner)
			
			# Normalize offset by eye width to be independent of distance from camera
			offset = (pupil - eye_center) / eye_width
			return offset
		except Exception:
			return None
	
	def _reset_session_state(self):
		# This function remains unchanged
		self.current_session_data = []; self.gaze_path.clear()
		self.fixations = 0; self.saccades = 0
		self.session_start_time = time.time()
		self.last_gaze_point = None; self.last_gaze_time = None
		self.is_in_fixation = False; self.fixation_start_time = None
		self.fixation_start_pos = None; self.counted_fixation = False
		self.kalman_initialized = False
	
	def _update_gaze_metrics(self, gaze_data: Dict[str, Any]):
		# This function remains unchanged
		current_time = gaze_data['timestamp']; current_point = np.array([gaze_data['x'], gaze_data['y']])
		if self.last_gaze_point is not None:
			dt = current_time - self.last_gaze_time
			if dt > 0:
				dist = np.linalg.norm(current_point - self.last_gaze_point); velocity = dist / dt
				if velocity < self.VELOCITY_THRESHOLD:
					if not self.is_in_fixation:
						self.is_in_fixation = True; self.fixation_start_time = current_time
						self.fixation_start_pos = current_point; self.counted_fixation = False
					elif not self.counted_fixation:
						fix_dist = np.linalg.norm(current_point - self.fixation_start_pos)
						fix_dur = current_time - self.fixation_start_time
						if fix_dur > self.FIXATION_DURATION_THRESHOLD and fix_dist < self.FIXATION_RADIUS_THRESHOLD:
							self.fixations += 1; self.counted_fixation = True
				else: 
					if self.is_in_fixation:
						self.saccades += 1; self.is_in_fixation = False; self.fixation_start_time = None
		self.last_gaze_point = current_point; self.last_gaze_time = current_time
		# Update adaptive thresholds from recent velocity history
		self._update_adaptive_thresholds()

	def _init_kalman(self, initial_x: float, initial_y: float):
		# State: [x, y, vx, vy]
		dt = 1/60.0
		self.k_F = np.array([[1, 0, dt, 0],
							[0, 1, 0, dt],
							[0, 0, 1, 0],
							[0, 0, 0, 1]], dtype=np.float32)
		self.k_H = np.array([[1, 0, 0, 0],
							[0, 1, 0, 0]], dtype=np.float32)
		self.k_P = np.eye(4, dtype=np.float32) * 1e3
		self.k_Q = np.eye(4, dtype=np.float32) * 1e-1
		self.k_R = np.eye(2, dtype=np.float32) * 5.0
		self.k_x = np.array([[initial_x], [initial_y], [0.0], [0.0]], dtype=np.float32)
		self.kalman_initialized = True

	def _kalman_update(self, meas_x: float, meas_y: float) -> Tuple[float, float]:
		if not self.kalman_initialized:
			self._init_kalman(meas_x, meas_y)
		# Predict
		self.k_x = self.k_F @ self.k_x
		self.k_P = self.k_F @ self.k_P @ self.k_F.T + self.k_Q
		# Update
		z = np.array([[meas_x], [meas_y]], dtype=np.float32)
		S = self.k_H @ self.k_P @ self.k_H.T + self.k_R
		K = self.k_P @ self.k_H.T @ np.linalg.inv(S)
		self.k_x = self.k_x + K @ (z - self.k_H @ self.k_x)
		self.k_P = (np.eye(4, dtype=np.float32) - K @ self.k_H) @ self.k_P
		return float(self.k_x[0,0]), float(self.k_x[1,0])

	def _update_adaptive_thresholds(self):
		# Adapt thresholds using recent velocities from gaze_path
		if len(self.gaze_path) < 10: return
		pts = np.array(self.gaze_path, dtype=np.float32)
		dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
		# approximate dt using 60 FPS
		vels = dists * 60.0
		if len(vels) == 0: return
		v_med = np.median(vels)
		v_iqr = np.subtract(*np.percentile(vels, [75, 25]))
		self.adaptive_velocity_threshold = float(np.clip(v_med + 3 * v_iqr, 500, 4000))
		self.adaptive_fix_radius = float(np.clip(np.percentile(dists, 25) + 10, 20, 80))
		self.adaptive_fix_dur = float(np.clip(self.FIXATION_DURATION_THRESHOLD, 0.1, 0.25))
	
	def run_live_screening(self):
		print(f"[START] STARTING LIVE SCREENING")
		if not self.is_trained: print("Models not trained."); return
	
		cam = cv2.VideoCapture(0)
		if not cam.isOpened(): print("[ERROR] CRITICAL ERROR: Cannot access webcam."); return
	
		cv2.namedWindow('Autism Screening', cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty('Autism Screening', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	
		# --- NEW: Calibration Phase ---
		while not self.is_calibrated:
			ret, frame = cam.read()
			if not ret: break
			
			calib_frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
			cv2.putText(calib_frame, "Look at the red circle", (int(self.screen_width/2)-250, int(self.screen_height/2)-50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
			cv2.putText(calib_frame, "Press 'C' to Calibrate", (int(self.screen_width/2)-280, int(self.screen_height/2)+100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
			cv2.circle(calib_frame, (int(self.screen_width/2), int(self.screen_height/2)), 30, (0, 0, 255), -1)
			cv2.imshow('Autism Screening', calib_frame)

			key = cv2.waitKey(1) & 0xFF
			if key == ord('c'):
				results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
				if results.multi_face_landmarks:
					landmarks = results.multi_face_landmarks[0].landmark
					offset = self._get_eye_offset(landmarks)
					if offset is not None:
						self.calibrated_gaze_offset = offset
						self.is_calibrated = True
						print("[SUCCESS] Calibration successful!")
					else:
						print("[ERROR] Calibration failed. Please try again.")
			elif key == ord('q'):
				cam.release(); cv2.destroyAllWindows(); return
		
		# --- Main Screening Phase ---
		ball_pos = np.array([self.screen_width / 2, self.screen_height / 2], dtype=float)
		ball_vel = np.array([7, 5], dtype=float); ball_radius = 40
		self._reset_session_state()
		
		try:
			while True:
				ret_cam, cam_frame = cam.read()
				if not ret_cam: print("Webcam disconnected."); break
				
				display_frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
				ball_pos += ball_vel
				if ball_pos[0]<=ball_radius or ball_pos[0]>=self.screen_width-ball_radius: ball_vel[0]*=-1
				if ball_pos[1]<=ball_radius or ball_pos[1]>=self.screen_height-ball_radius: ball_vel[1]*=-1
				cv2.circle(display_frame, tuple(ball_pos.astype(int)), ball_radius, (255, 255, 255), -1)
				
				flipped_cam_frame = cv2.flip(cam_frame, 1)
				results = self.face_mesh.process(cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB))
				gaze_data = None
				
				if results.multi_face_landmarks:
					landmarks = results.multi_face_landmarks[0].landmark
					current_offset = self._get_eye_offset(landmarks)
					
					if current_offset is not None:
						# Calculate gaze deviation from calibrated center
						gaze_deviation = current_offset - self.calibrated_gaze_offset
						
						# Map deviation to screen coordinates (raw calculation)
						raw_screen_x = self.screen_width / 2 - gaze_deviation[0] * self.screen_width * self.GAZE_SENSITIVITY
						raw_screen_y = self.screen_height / 2 + gaze_deviation[1] * self.screen_height * self.GAZE_SENSITIVITY
						# Apply Kalman smoothing (with EMA fallback)
						k_x, k_y = self._kalman_update(raw_screen_x, raw_screen_y)
						smoothed_x = (self.last_smoothed_gaze[0] * self.SMOOTHING_FACTOR) + (k_x * (1 - self.SMOOTHING_FACTOR))
						smoothed_y = (self.last_smoothed_gaze[1] * self.SMOOTHING_FACTOR) + (k_y * (1 - self.SMOOTHING_FACTOR))
						self.last_smoothed_gaze = np.array([smoothed_x, smoothed_y])

						# Clamp final coordinates to stay within screen bounds
						screen_x = np.clip(smoothed_x, 0, self.screen_width)
						screen_y = np.clip(smoothed_y, 0, self.screen_height)

						gaze_data = {'x': screen_x, 'y': screen_y, 'timestamp': time.time()}

				if gaze_data:
					self._update_gaze_metrics(gaze_data)
					self.current_session_data.append(gaze_data)
					self.gaze_path.append((int(gaze_data['x']), int(gaze_data['y'])))
					
					if len(self.gaze_path) > 2:
						path_points = np.array(self.gaze_path, dtype=np.int32).reshape((-1, 1, 2))
						cv2.polylines(display_frame, [path_points], isClosed=False, color=(0, 0, 255), thickness=3)
					
					overlay = display_frame.copy()
					cv2.circle(overlay, (int(gaze_data['x']), int(gaze_data['y'])), 20, (0, 255, 0), -1)
					display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)
				
				# Eyecam display code remains the same...
				facecam_w, facecam_h, margin = 320, 240, 20
				eyecam_view = cv2.resize(flipped_cam_frame, (facecam_w, facecam_h))
				if results.multi_face_landmarks:
					landmarks = results.multi_face_landmarks[0].landmark
					LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
					h, w, _ = flipped_cam_frame.shape
					eye_points = np.array([(landmarks[i].x*w, landmarks[i].y*h) for i in LEFT_EYE_CONTOUR], dtype=np.int32)
					x_min, y_min = np.min(eye_points, axis=0); x_max, y_max = np.max(eye_points, axis=0)
					padding = 25
					x_min, y_min = max(0, x_min-padding), max(0, y_min-padding)
					x_max, y_max = min(w, x_max+padding), min(h, y_max+padding)
					if x_max > x_min and y_max > y_min:
						eye_crop = flipped_cam_frame[y_min:y_max, x_min:x_max]
						# Add pink-purple markers to eye landmarks
						for point in eye_points:
							px = int(point[0] - x_min)
							py = int(point[1] - y_min)
							cv2.circle(eye_crop, (px, py), 3, (255, 0, 255), -1)
						eyecam_view = cv2.resize(eye_crop, (facecam_w, facecam_h))
				roi_x1 = self.screen_width - facecam_w - margin; roi_y1 = margin
				display_frame[roi_y1:(roi_y1+facecam_h), roi_x1:(roi_x1+facecam_w)] = eyecam_view
				
				# Displaying metrics and exit text remains the same...
				elapsed_time = time.time()-self.session_start_time
				fix_sacc_ratio = self.fixations / self.saccades if self.saccades > 0 else self.fixations * 1000.0
				metrics = [f"Time: {elapsed_time:.1f}s", f"Fixations: {self.fixations}", f"Saccades: {self.saccades}", f"Fix/Sacc Ratio: {fix_sacc_ratio:.2f}"]
				for i, text in enumerate(metrics): cv2.putText(display_frame, text, (30, 60 + i * 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
				exit_text = "Press Q to Exit"
				text_size = cv2.getTextSize(exit_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
				cv2.putText(display_frame, exit_text, (self.screen_width-text_size[0]-20, self.screen_height-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
				
				cv2.imshow('Autism Screening', display_frame)
				if cv2.waitKey(1) & 0xFF == ord('q') or elapsed_time >= 60: break
		finally:
			cam.release(); cv2.destroyAllWindows()
			# Export session data
			if len(self.current_session_data) > 0:
				try:
					df = pd.DataFrame(self.current_session_data)
					ts = time.strftime('%Y%m%d_%H%M%S')
					out_path = SCRIPT_DIR / f"session_{ts}.csv"
					df.to_csv(out_path, index=False)
					print(f"[INFO] Session data saved to {out_path}")
				except Exception as e:
					print(f"[WARN] Could not save session CSV: {e}")
			if len(self.current_session_data) > 50: self.generate_final_prediction()
		
	# Functions generate_final_prediction and generate_visual_report remain unchanged...
	def generate_final_prediction(self):
		df = pd.DataFrame(self.current_session_data); features = self.extract_comprehensive_features(df)
		if not features: return
		# Model-based prediction with stacked ensemble and uncertainty metrics
		print("\n--- SCREENING RESULT ---")
		# Compute model_probs for report visualization
		print("------------------------\n")
		f_vector = np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)
		f_vector_s = self.scaler.transform(f_vector)
		model_probs = {}
		for name, m_data in self.ml_models.items(): model_probs[name] = m_data['model'].predict_proba(f_vector_s)[0, 1]
		model_probs['DNN'] = self.dl_models['DNN']['model'].predict(f_vector_s, verbose=0)[0, 0]
		# Stacked meta prediction
		stack_vec = np.array([[model_probs.get('RF', 0.0), model_probs.get('SVM', 0.0), model_probs.get('DNN', 0.0)]])
		meta = LogisticRegression(max_iter=1000)
		# Reuse trained meta if available via ensemble coefficients; otherwise fit a quick calibrator on RF if needed
		try:
			if self.ensemble_model and self.ensemble_model.get('type') == 'stacked':
				coef = np.array(self.ensemble_model['coefficients'])
				inter = np.array(self.ensemble_model['intercept'])
				# Build a temporary LogisticRegression-ish scorer
				logit = float(stack_vec @ coef.T + inter)
				meta_prob = 1.0 / (1.0 + np.exp(-logit))
			else:
				meta_prob = float(np.mean(list(model_probs.values())))
		except Exception:
			meta_prob = float(np.mean(list(model_probs.values())))
		probs_list = list(model_probs.values()) + [meta_prob]
		pred_std = float(np.std(probs_list))
		pred_entropy = float(-(meta_prob*np.log2(max(meta_prob,1e-9)) + (1-meta_prob)*np.log2(max(1-meta_prob,1e-9))))
		verdict = "Autistic Syndrome" if meta_prob >= 0.65 else "Not Autistic"
		print(f"Final Verdict: {verdict}")
		print(f"ASD Probability (stacked): {meta_prob:.2%}")
		print(f"Uncertainty (std across models): {pred_std:.3f} | Entropy: {pred_entropy:.3f} bits")
		self.generate_visual_report(df, model_probs, verdict, meta_prob, pred_std, pred_entropy)
	
	def generate_visual_report(self, df: pd.DataFrame, model_probs: Dict[str, float], verdict: str, asd_prob: float = None, prob_std: float = None, prob_entropy_bits: float = None):
		print("Generating visual report...")
		dx=np.diff(df['x'].values); dy=np.diff(df['y'].values); dt=np.diff(df['timestamp'].values)
		dt[dt==0] = 1e-6; velocities = np.sqrt(dx*2 + dy*2) / dt
		plt.style.use('dark_background'); fig = plt.figure(figsize=(18, 10))
		fig.suptitle(f'Autism Screening Analysis - Final Verdict: {verdict}', fontsize=20, color='lightgray')
		ax1=plt.subplot(2,3,1); ax1.plot(df['x'],df['y'],color='red',alpha=0.7); ax1.scatter(df['x'].iloc[0],df['y'].iloc[0],c='lime',s=100,label='Start'); ax1.scatter(df['x'].iloc[-1],df['y'].iloc[-1],c='cyan',s=100,label='End'); ax1.set_xlim(0,self.screen_width); ax1.set_ylim(self.screen_height,0); ax1.set_title('Gaze Scan Path',color='white'); ax1.set_aspect('equal',adjustable='box'); ax1.legend()
		ax2=plt.subplot(2,3,2); ax2.plot(df['timestamp'].iloc[1:]-df['timestamp'].iloc[0],velocities,color='orange'); ax2.axhline(y=self.VELOCITY_THRESHOLD,color='cyan',linestyle='--',label=f'Saccade Threshold ({self.VELOCITY_THRESHOLD} px/s)'); ax2.set_title('Gaze Velocity Over Time',color='white'); ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Velocity (pixels/sec)'); ax2.legend()
		ax3=plt.subplot(2,3,3); events=['Fixations','Saccades']; counts=[self.fixations,self.saccades]; ax3.bar(events,counts,color=['green','red']); ax3.set_title('Fixation & Saccade Event Counts',color='white'); ax3.set_ylabel('Total Count')
		ax4=plt.subplot(2,3,4); sns.kdeplot(x=df['x'],y=df['y'],cmap="rocket",fill=True,thresh=0.05,ax=ax4); ax4.set_xlim(0,self.screen_width); ax4.set_ylim(self.screen_height,0); ax4.set_title('Gaze Point Heatmap',color='white'); ax4.set_aspect('equal',adjustable='box')
		ax5=plt.subplot(2,3,5); models=list(model_probs.keys()); probs=list(model_probs.values()); ax5.bar(models,probs,color='lightblue'); ax5.axhline(y=0.65,color='red',linestyle='--',label='ASD Threshold (0.65)'); ax5.set_ylim(0,1); ax5.set_title('Individual Model Predictions',color='white'); ax5.set_ylabel('ASD Probability'); ax5.legend()
		# Annotate uncertainty and stacked probability
		if asd_prob is not None:
			ci_low = max(0.0, asd_prob - (prob_std if prob_std is not None else 0.0))
			ci_high = min(1.0, asd_prob + (prob_std if prob_std is not None else 0.0))
			info = f"Stacked ASD P: {asd_prob:.2%}\nUncertainty (std): {prob_std:.3f}\nEntropy: {prob_entropy_bits:.3f} bits\n~CI: [{ci_low:.2f}, {ci_high:.2f}]"
			fig.text(0.67, 0.06, info, fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.3, boxstyle='round'))
		# Add learning curves if available
		if 'DNN' in self.dl_models and 'history' in self.dl_models['DNN']:
			ax6=plt.subplot(2,3,6)
			hist = self.dl_models['DNN']['history']
			if 'auc' in hist and 'val_auc' in hist:
				ax6.plot(hist['auc'], label='Train AUC')
				ax6.plot(hist['val_auc'], label='Val AUC')
				ax6.set_title('DNN Learning Curve (AUC)', color='white')
				ax6.legend()
		plt.tight_layout(rect=[0,0,1,0.96]); report_path = SCRIPT_DIR / "Screening_Report.png"
		plt.savefig(report_path); print(f"Report saved to {report_path}"); plt.show()

if _name_ == "_main_":
	TRAINING_DATA_CSV = SCRIPT_DIR / "srijan_features_only_with_groups.csv"
	system = AutismScreeningSystem(csv_path=str(TRAINING_DATA_CSV))
	if not system.load_models():
		print(" No pre-trained models found. Training new models...")
		system.train_all_models()
	if system.is_trained:
		system.run_live_screening()
	print("Program finished.")