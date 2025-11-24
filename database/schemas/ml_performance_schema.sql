-- ================================================================
-- COMPREHENSIVE ML MODEL PERFORMANCE TRACKING SCHEMA
-- Underground Utility Detection Platform - Advanced Model Analytics
-- ================================================================

-- Model categories and algorithms registry
CREATE TABLE model_categories (
    id SERIAL PRIMARY KEY,
    category_code VARCHAR(20) UNIQUE NOT NULL,
    category_name VARCHAR(100) NOT NULL,
    description TEXT,
    typical_use_cases TEXT[],
    evaluation_metrics TEXT[] -- Relevant metrics for this category
);

-- Insert model categories
INSERT INTO model_categories (category_code, category_name, description, typical_use_cases, evaluation_metrics) VALUES
('CLASSIFICATION', 'Image Classification', 'GPR image classification models', ARRAY['cavity_detection', 'utility_identification', 'soil_type_classification'], ARRAY['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']),
('DETECTION', 'Object Detection', 'Utility object detection in GPR images', ARRAY['utility_localization', 'multi_utility_detection'], ARRAY['map_50', 'map_75', 'precision', 'recall', 'iou']),
('REGRESSION', 'Depth Estimation', 'Utility depth estimation models', ARRAY['depth_prediction', 'velocity_estimation'], ARRAY['mse', 'rmse', 'mae', 'r_squared', 'mape']),
('SEGMENTATION', 'Image Segmentation', 'Pixel-wise utility segmentation', ARRAY['utility_boundary_detection', 'hyperbola_segmentation'], ARRAY['iou', 'dice_coefficient', 'pixel_accuracy']),
('TIMESERIES', 'Signal Analysis', 'Time-series GPR signal analysis', ARRAY['signal_classification', 'anomaly_detection'], ARRAY['accuracy', 'precision', 'recall', 'auc_pr']),
('ENSEMBLE', 'Ensemble Models', 'Combined model approaches', ARRAY['multi_modal_fusion', 'voting_classifiers'], ARRAY['accuracy', 'ensemble_diversity', 'individual_performance']);

-- Model architectures registry
CREATE TABLE model_architectures (
    id SERIAL PRIMARY KEY,
    architecture_code VARCHAR(50) UNIQUE NOT NULL,
    architecture_name VARCHAR(100) NOT NULL,
    category_id INTEGER NOT NULL REFERENCES model_categories(id),
    framework VARCHAR(50), -- tensorflow, pytorch, sklearn, etc.
    typical_parameters JSONB,
    computational_complexity VARCHAR(20), -- low, medium, high, very_high
    memory_requirements VARCHAR(20),
    training_time_estimate VARCHAR(50)
);

-- Insert common architectures
INSERT INTO model_architectures (architecture_code, architecture_name, category_id, framework, computational_complexity, memory_requirements) VALUES
('RESNET50', 'ResNet-50', 1, 'tensorflow', 'medium', 'medium'),
('EFFICIENTNET', 'EfficientNet', 1, 'tensorflow', 'medium', 'low'),
('VGG16', 'VGG-16', 1, 'tensorflow', 'high', 'high'),
('YOLOV8', 'YOLOv8', 2, 'pytorch', 'high', 'medium'),
('FASTER_RCNN', 'Faster R-CNN', 2, 'tensorflow', 'very_high', 'high'),
('UNET', 'U-Net', 4, 'tensorflow', 'medium', 'medium'),
('RANDOM_FOREST', 'Random Forest', 1, 'sklearn', 'low', 'low'),
('XGBOOST', 'XGBoost', 3, 'xgboost', 'medium', 'low'),
('LSTM', 'LSTM', 5, 'tensorflow', 'medium', 'medium'),
('TRANSFORMER', 'Transformer', 5, 'pytorch', 'high', 'high');

-- ================================================================
-- ENHANCED ML MODELS REGISTRY
-- ================================================================

CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,

    -- Model categorization
    category_id INTEGER NOT NULL REFERENCES model_categories(id),
    architecture_id INTEGER NOT NULL REFERENCES model_architectures(id),

    -- Model purpose and scope
    primary_objective TEXT NOT NULL,
    secondary_objectives TEXT[],
    target_datasets TEXT[], -- Twente, Mojahid, etc.
    application_domain VARCHAR(100), -- utility_detection, depth_estimation, etc.

    -- Technical specifications
    input_specifications JSONB, -- Input shape, preprocessing requirements
    output_specifications JSONB, -- Output format, post-processing
    hyperparameters JSONB, -- Model hyperparameters
    preprocessing_pipeline JSONB, -- Data preprocessing steps

    -- Model complexity metrics
    total_parameters BIGINT,
    trainable_parameters BIGINT,
    model_size_mb DECIMAL(10,2),
    flops_count BIGINT, -- Floating point operations

    -- Training configuration
    training_framework VARCHAR(50),
    optimizer_config JSONB,
    loss_function VARCHAR(100),
    learning_rate_schedule JSONB,
    regularization_techniques TEXT[],

    -- Training resources
    training_hardware VARCHAR(200),
    training_duration_hours DECIMAL(8,2),
    gpu_memory_used_gb DECIMAL(6,2),
    training_cost_estimate DECIMAL(10,2),

    -- Data specifications
    training_dataset_size INTEGER,
    validation_dataset_size INTEGER,
    test_dataset_size INTEGER,
    data_augmentation_applied BOOLEAN DEFAULT FALSE,
    augmentation_techniques TEXT[],

    -- Version control and reproducibility
    code_repository_url TEXT,
    code_commit_hash VARCHAR(64),
    random_seed INTEGER,
    environment_specification JSONB, -- Python packages, versions

    -- Model files and artifacts
    model_file_path TEXT,
    weights_file_path TEXT,
    config_file_path TEXT,
    training_logs_path TEXT,
    tensorboard_logs_path TEXT,

    -- Status and lifecycle
    development_stage VARCHAR(50) DEFAULT 'development' CHECK (development_stage IN ('development', 'training', 'validation', 'testing', 'production', 'deprecated')),
    deployment_status VARCHAR(50) DEFAULT 'not_deployed' CHECK (deployment_status IN ('not_deployed', 'staging', 'production', 'retired')),

    -- Metadata
    created_by VARCHAR(255),
    model_description TEXT,
    training_notes TEXT,
    known_limitations TEXT[],
    recommended_use_cases TEXT[],

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(model_name, model_version)
);

-- ================================================================
-- COMPREHENSIVE CROSS-VALIDATION TRACKING
-- ================================================================

CREATE TABLE cross_validation_experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,

    -- Experiment identification
    experiment_name VARCHAR(255) NOT NULL,
    experiment_description TEXT,

    -- Cross-validation configuration
    cv_strategy VARCHAR(50) NOT NULL, -- k_fold, stratified_k_fold, time_series_split, etc.
    n_folds INTEGER NOT NULL CHECK (n_folds >= 2),
    random_state INTEGER,
    shuffle_data BOOLEAN DEFAULT TRUE,
    stratification_column VARCHAR(100), -- For stratified CV

    -- Dataset configuration
    dataset_identifier VARCHAR(255) NOT NULL,
    total_samples INTEGER NOT NULL,
    feature_columns TEXT[],
    target_column VARCHAR(100),

    -- Experiment timing
    experiment_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    experiment_end TIMESTAMP WITH TIME ZONE,
    total_duration_minutes DECIMAL(8,2),

    -- Overall results aggregation
    mean_accuracy DECIMAL(6,4),
    std_accuracy DECIMAL(6,4),
    mean_precision DECIMAL(6,4),
    std_precision DECIMAL(6,4),
    mean_recall DECIMAL(6,4),
    std_recall DECIMAL(6,4),
    mean_f1_score DECIMAL(6,4),
    std_f1_score DECIMAL(6,4),

    -- Additional metrics based on model type
    mean_auc_roc DECIMAL(6,4),
    std_auc_roc DECIMAL(6,4),
    mean_auc_pr DECIMAL(6,4),
    std_auc_pr DECIMAL(6,4),

    -- Regression metrics (if applicable)
    mean_mse DECIMAL(10,6),
    std_mse DECIMAL(10,6),
    mean_rmse DECIMAL(10,6),
    std_rmse DECIMAL(10,6),
    mean_mae DECIMAL(10,6),
    std_mae DECIMAL(10,6),
    mean_r_squared DECIMAL(6,4),
    std_r_squared DECIMAL(6,4),

    -- Cross-validation stability metrics
    performance_stability_score DECIMAL(5,4), -- Low std = high stability
    worst_fold_performance DECIMAL(6,4),
    best_fold_performance DECIMAL(6,4),
    performance_range DECIMAL(6,4),

    -- Statistical significance
    confidence_interval_95_lower DECIMAL(6,4),
    confidence_interval_95_upper DECIMAL(6,4),
    p_value_significance DECIMAL(8,6),

    -- Computational resources
    average_fold_training_time_minutes DECIMAL(8,2),
    total_training_time_minutes DECIMAL(8,2),
    memory_usage_peak_gb DECIMAL(6,2),

    -- Experiment status
    status VARCHAR(50) DEFAULT 'running' CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    error_message TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Individual fold results
CREATE TABLE cv_fold_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cv_experiment_id UUID NOT NULL REFERENCES cross_validation_experiments(id) ON DELETE CASCADE,

    -- Fold identification
    fold_number INTEGER NOT NULL CHECK (fold_number >= 0),
    train_indices INTEGER[], -- Indices of training samples
    validation_indices INTEGER[], -- Indices of validation samples

    -- Fold dataset characteristics
    train_size INTEGER NOT NULL,
    validation_size INTEGER NOT NULL,
    class_distribution_train JSONB, -- For classification tasks
    class_distribution_validation JSONB,

    -- Training details
    training_start TIMESTAMP WITH TIME ZONE,
    training_end TIMESTAMP WITH TIME ZONE,
    training_duration_minutes DECIMAL(8,2),
    training_epochs INTEGER,
    early_stopping_epoch INTEGER,

    -- Performance metrics
    validation_accuracy DECIMAL(6,4),
    validation_precision DECIMAL(6,4),
    validation_recall DECIMAL(6,4),
    validation_f1_score DECIMAL(6,4),
    validation_auc_roc DECIMAL(6,4),
    validation_auc_pr DECIMAL(6,4),

    -- Regression metrics
    validation_mse DECIMAL(10,6),
    validation_rmse DECIMAL(10,6),
    validation_mae DECIMAL(10,6),
    validation_r_squared DECIMAL(6,4),
    validation_mape DECIMAL(6,4),

    -- Per-class metrics (for multi-class problems)
    per_class_precision JSONB,
    per_class_recall JSONB,
    per_class_f1_score JSONB,

    -- Confusion matrix and predictions
    confusion_matrix JSONB,
    prediction_probabilities JSONB, -- Stored selectively for analysis
    feature_importance JSONB, -- If available

    -- Training monitoring
    training_loss_history JSONB,
    validation_loss_history JSONB,
    learning_rate_history JSONB,

    -- Resource usage
    memory_usage_mb DECIMAL(8,2),
    gpu_utilization_percent DECIMAL(5,2),

    -- Quality indicators
    overfitting_detected BOOLEAN DEFAULT FALSE,
    underfitting_detected BOOLEAN DEFAULT FALSE,
    convergence_achieved BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- MODEL PERFORMANCE MONITORING
-- ================================================================

CREATE TABLE model_performance_monitoring (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,

    -- Monitoring period
    monitoring_start TIMESTAMP WITH TIME ZONE NOT NULL,
    monitoring_end TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Dataset information
    evaluation_dataset VARCHAR(255) NOT NULL,
    dataset_version VARCHAR(50),
    total_predictions INTEGER NOT NULL,

    -- Performance drift detection
    baseline_accuracy DECIMAL(6,4),
    current_accuracy DECIMAL(6,4),
    accuracy_drift DECIMAL(6,4), -- Difference from baseline
    drift_significance VARCHAR(20), -- negligible, minor, moderate, major, critical

    -- Prediction distribution analysis
    prediction_distribution JSONB,
    confidence_distribution JSONB,
    prediction_entropy DECIMAL(8,4),

    -- Error analysis
    systematic_errors_detected TEXT[],
    error_pattern_analysis JSONB,
    misclassification_patterns JSONB,

    -- Data quality indicators
    data_quality_score DECIMAL(5,4),
    outlier_detection_rate DECIMAL(5,4),
    missing_value_rate DECIMAL(5,4),
    distribution_shift_detected BOOLEAN DEFAULT FALSE,

    -- Performance by subgroups
    performance_by_utility_type JSONB,
    performance_by_depth_range JSONB,
    performance_by_ground_conditions JSONB,

    -- Computational performance
    average_inference_time_ms DECIMAL(8,3),
    throughput_predictions_per_second DECIMAL(8,2),
    memory_usage_inference_mb DECIMAL(8,2),

    -- Alerting and notifications
    performance_alerts TEXT[],
    recommended_actions TEXT[],
    retraining_recommended BOOLEAN DEFAULT FALSE,
    model_retirement_suggested BOOLEAN DEFAULT FALSE,

    -- Monitoring metadata
    monitoring_algorithm VARCHAR(100),
    statistical_tests_applied TEXT[],
    analyst_notes TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- FEATURE IMPORTANCE AND ANALYSIS
-- ================================================================

CREATE TABLE feature_importance_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,
    cv_experiment_id UUID REFERENCES cross_validation_experiments(id),

    -- Analysis configuration
    importance_method VARCHAR(100) NOT NULL, -- permutation, shap, lime, etc.
    analysis_dataset VARCHAR(255) NOT NULL,
    sample_size INTEGER,

    -- Global feature importance
    feature_importance_scores JSONB NOT NULL, -- {feature_name: importance_score}
    feature_importance_ranking JSONB, -- Ranked list of features

    -- Statistical significance of features
    feature_p_values JSONB,
    significant_features_001 TEXT[], -- p < 0.001
    significant_features_005 TEXT[], -- p < 0.05
    significant_features_010 TEXT[], -- p < 0.10

    -- Feature interaction analysis
    top_feature_interactions JSONB, -- Important feature pairs/triplets
    interaction_strength_scores JSONB,

    -- Stability analysis across CV folds
    feature_importance_stability JSONB,
    consistently_important_features TEXT[],
    unstable_features TEXT[],

    -- Domain-specific insights
    signal_features_importance JSONB, -- GPR signal features
    environmental_features_importance JSONB, -- Environmental metadata
    spatial_features_importance JSONB, -- Spatial/geometric features

    -- Dimensionality insights
    cumulative_importance_curve JSONB,
    importance_threshold_50_percent INTEGER, -- Number of features for 50% importance
    importance_threshold_80_percent INTEGER, -- Number of features for 80% importance
    importance_threshold_95_percent INTEGER, -- Number of features for 95% importance

    -- Feature engineering recommendations
    redundant_features TEXT[],
    highly_correlated_feature_pairs JSONB,
    suggested_feature_combinations TEXT[],
    feature_engineering_opportunities TEXT[],

    -- Analysis metadata
    analysis_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    computation_time_minutes DECIMAL(8,2),
    analyst_notes TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- MODEL COMPARISON AND BENCHMARKING
-- ================================================================

CREATE TABLE model_comparison_experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Comparison identification
    comparison_name VARCHAR(255) NOT NULL,
    comparison_description TEXT,
    comparison_objective TEXT NOT NULL,

    -- Experimental setup
    models_compared UUID[] NOT NULL, -- Array of model IDs
    evaluation_datasets TEXT[] NOT NULL,
    evaluation_metrics TEXT[] NOT NULL,
    statistical_test VARCHAR(100), -- t-test, wilcoxon, etc.
    significance_level DECIMAL(3,2) DEFAULT 0.05,

    -- Comparison results
    best_performing_model_id UUID REFERENCES ml_models(id),
    performance_ranking JSONB, -- Ranked model performance
    statistical_significance_matrix JSONB, -- Pairwise significance tests

    -- Detailed metrics comparison
    accuracy_comparison JSONB,
    precision_comparison JSONB,
    recall_comparison JSONB,
    f1_score_comparison JSONB,
    auc_roc_comparison JSONB,

    -- Efficiency comparison
    training_time_comparison JSONB,
    inference_time_comparison JSONB,
    memory_usage_comparison JSONB,
    model_size_comparison JSONB,

    -- Robustness analysis
    cross_validation_stability_comparison JSONB,
    performance_variance_comparison JSONB,
    worst_case_performance_comparison JSONB,

    -- Domain-specific comparison
    performance_by_utility_type_comparison JSONB,
    performance_by_depth_comparison JSONB,
    performance_by_conditions_comparison JSONB,

    -- Overall recommendations
    recommended_model_id UUID REFERENCES ml_models(id),
    recommendation_reasoning TEXT,
    trade_off_analysis TEXT,
    deployment_recommendations TEXT[],

    -- Experiment metadata
    conducted_by VARCHAR(255),
    experiment_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    review_status VARCHAR(50) DEFAULT 'pending' CHECK (review_status IN ('pending', 'reviewed', 'approved', 'rejected')),
    reviewer_notes TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ================================================================
-- HYPERPARAMETER OPTIMIZATION TRACKING
-- ================================================================

CREATE TABLE hyperparameter_optimization (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,

    -- Optimization configuration
    optimization_algorithm VARCHAR(100) NOT NULL, -- grid_search, random_search, bayesian, etc.
    parameter_space JSONB NOT NULL, -- Search space definition
    optimization_objective VARCHAR(100) NOT NULL, -- accuracy, f1_score, etc.
    optimization_direction VARCHAR(10) CHECK (optimization_direction IN ('maximize', 'minimize')),

    -- Search configuration
    max_evaluations INTEGER,
    timeout_minutes INTEGER,
    n_jobs INTEGER DEFAULT 1,
    cv_folds INTEGER DEFAULT 5,

    -- Best results
    best_parameters JSONB,
    best_score DECIMAL(8,6),
    best_trial_number INTEGER,

    -- Optimization history
    trial_history JSONB, -- All trials and their results
    convergence_history JSONB, -- Best score over time

    -- Search efficiency
    total_trials INTEGER,
    successful_trials INTEGER,
    failed_trials INTEGER,
    total_optimization_time_hours DECIMAL(8,2),

    -- Analysis insights
    parameter_importance JSONB,
    parameter_correlations JSONB,
    optimization_insights TEXT[],

    -- Status
    optimization_status VARCHAR(50) DEFAULT 'running' CHECK (optimization_status IN ('queued', 'running', 'completed', 'failed', 'stopped')),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- ================================================================
-- PERFORMANCE INDEXES
-- ================================================================

-- ML models indexes
CREATE INDEX idx_ml_models_category ON ml_models(category_id);
CREATE INDEX idx_ml_models_architecture ON ml_models(architecture_id);
CREATE INDEX idx_ml_models_status ON ml_models(development_stage, deployment_status);
CREATE INDEX idx_ml_models_created ON ml_models(created_at);

-- Cross-validation indexes
CREATE INDEX idx_cv_experiments_model ON cross_validation_experiments(model_id);
CREATE INDEX idx_cv_experiments_dataset ON cross_validation_experiments(dataset_identifier);
CREATE INDEX idx_cv_experiments_status ON cross_validation_experiments(status);
CREATE INDEX idx_cv_fold_results_experiment ON cv_fold_results(cv_experiment_id);
CREATE INDEX idx_cv_fold_results_fold ON cv_fold_results(fold_number);

-- Performance monitoring indexes
CREATE INDEX idx_performance_monitoring_model ON model_performance_monitoring(model_id);
CREATE INDEX idx_performance_monitoring_period ON model_performance_monitoring(monitoring_start, monitoring_end);
CREATE INDEX idx_performance_monitoring_drift ON model_performance_monitoring(accuracy_drift);

-- Feature importance indexes
CREATE INDEX idx_feature_importance_model ON feature_importance_analysis(model_id);
CREATE INDEX idx_feature_importance_experiment ON feature_importance_analysis(cv_experiment_id);
CREATE INDEX idx_feature_importance_method ON feature_importance_analysis(importance_method);

-- Model comparison indexes
CREATE INDEX idx_model_comparison_models ON model_comparison_experiments USING GIN(models_compared);
CREATE INDEX idx_model_comparison_best ON model_comparison_experiments(best_performing_model_id);

-- Hyperparameter optimization indexes
CREATE INDEX idx_hyperopt_model ON hyperparameter_optimization(model_id);
CREATE INDEX idx_hyperopt_algorithm ON hyperparameter_optimization(optimization_algorithm);
CREATE INDEX idx_hyperopt_status ON hyperparameter_optimization(optimization_status);

-- ================================================================
-- MATERIALIZED VIEWS FOR ML ANALYTICS
-- ================================================================

-- Model performance leaderboard
CREATE MATERIALIZED VIEW model_performance_leaderboard AS
SELECT
    m.id as model_id,
    m.model_name,
    m.model_version,
    mc.category_name,
    ma.architecture_name,
    cv.mean_accuracy,
    cv.std_accuracy,
    cv.mean_f1_score,
    cv.performance_stability_score,
    m.training_duration_hours,
    m.model_size_mb,
    m.total_parameters,
    m.deployment_status,
    ROW_NUMBER() OVER (PARTITION BY mc.category_name ORDER BY cv.mean_f1_score DESC) as category_rank,
    ROW_NUMBER() OVER (ORDER BY cv.mean_f1_score DESC) as overall_rank
FROM ml_models m
JOIN model_categories mc ON m.category_id = mc.id
JOIN model_architectures ma ON m.architecture_id = ma.id
LEFT JOIN cross_validation_experiments cv ON m.id = cv.model_id
WHERE cv.status = 'completed'
ORDER BY cv.mean_f1_score DESC;

-- Model efficiency analysis
CREATE MATERIALIZED VIEW model_efficiency_analysis AS
SELECT
    m.id as model_id,
    m.model_name,
    mc.category_name,
    cv.mean_f1_score as performance_score,
    m.training_duration_hours,
    m.model_size_mb,
    m.total_parameters,
    ROUND(cv.mean_f1_score / NULLIF(m.training_duration_hours, 0), 6) as performance_per_hour,
    ROUND(cv.mean_f1_score / NULLIF(m.model_size_mb, 0), 6) as performance_per_mb,
    ROUND(cv.mean_f1_score / NULLIF(m.total_parameters::DECIMAL / 1000000, 0), 6) as performance_per_million_params
FROM ml_models m
JOIN model_categories mc ON m.category_id = mc.id
LEFT JOIN cross_validation_experiments cv ON m.id = cv.model_id
WHERE cv.status = 'completed'
ORDER BY performance_per_hour DESC;

-- ================================================================
-- UTILITY FUNCTIONS
-- ================================================================

-- Function to calculate model complexity score
CREATE OR REPLACE FUNCTION calculate_model_complexity_score(model_uuid UUID)
RETURNS DECIMAL AS $$
DECLARE
    model_record RECORD;
    complexity_score DECIMAL;
BEGIN
    SELECT * INTO model_record FROM ml_models WHERE id = model_uuid;

    IF NOT FOUND THEN
        RETURN NULL;
    END IF;

    -- Normalize parameters (log scale), size, and training time
    complexity_score :=
        LOG(GREATEST(model_record.total_parameters, 1)) * 0.4 +
        LOG(GREATEST(model_record.model_size_mb, 0.1)) * 0.3 +
        LOG(GREATEST(model_record.training_duration_hours, 0.1)) * 0.3;

    RETURN ROUND(complexity_score, 4);
END;
$$ LANGUAGE plpgsql;

-- Function to refresh ML analytics views
CREATE OR REPLACE FUNCTION refresh_ml_analytics_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW model_performance_leaderboard;
    REFRESH MATERIALIZED VIEW model_efficiency_analysis;
END;
$$ LANGUAGE plpgsql;

-- ================================================================
-- COMMENTS FOR DOCUMENTATION
-- ================================================================

COMMENT ON TABLE ml_models IS 'Comprehensive ML model registry with detailed specifications and lifecycle management';
COMMENT ON TABLE cross_validation_experiments IS 'Detailed cross-validation experiment tracking with statistical analysis';
COMMENT ON TABLE cv_fold_results IS 'Individual fold results for comprehensive cross-validation analysis';
COMMENT ON TABLE model_performance_monitoring IS 'Continuous model performance monitoring and drift detection';
COMMENT ON TABLE feature_importance_analysis IS 'Feature importance analysis with stability and interaction insights';
COMMENT ON TABLE model_comparison_experiments IS 'Systematic model comparison and benchmarking experiments';
COMMENT ON TABLE hyperparameter_optimization IS 'Hyperparameter optimization tracking and analysis';

COMMENT ON MATERIALIZED VIEW model_performance_leaderboard IS 'Performance ranking of models across categories';
COMMENT ON MATERIALIZED VIEW model_efficiency_analysis IS 'Model efficiency metrics balancing performance and computational cost';