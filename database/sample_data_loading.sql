-- ================================================================
-- UNDERGROUND UTILITY DETECTION PLATFORM - SAMPLE DATA LOADING
-- Comprehensive Sample Data for All Datasets
-- ================================================================

-- Enable necessary extensions and check dependencies
DO $$
BEGIN
    -- Verify required tables exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'projects') THEN
        RAISE EXCEPTION 'Core schema tables not found. Please run schema deployment scripts first.';
    END IF;

    RAISE NOTICE 'Starting sample data loading process...';
END $$;

-- ================================================================
-- UNIVERSITY OF TWENTE GPR DATASET SAMPLE DATA
-- 25+ metadata fields from real GPR surveys
-- ================================================================

-- Create sample project for Twente dataset
INSERT INTO projects (id, name, description, client_organization, project_manager, start_date, end_date, compliance_standards)
VALUES
    (uuid_generate_v4(), 'Twente University GPR Research Dataset',
     'Ground-truthed GPR dataset from 13 construction sites in Netherlands with comprehensive environmental metadata',
     'University of Twente', 'Dr. GPR Research Lead', '2023-01-15', '2023-12-20',
     ARRAY['PAS128:2022', 'ISO 14688'])
ON CONFLICT DO NOTHING;

-- Get the project ID for reference
DO $$
DECLARE
    twente_project_id UUID;
BEGIN
    SELECT id INTO twente_project_id FROM projects WHERE name = 'Twente University GPR Research Dataset';

    -- Create survey sites based on Twente locations
    INSERT INTO survey_sites (id, project_id, site_name, address, location, land_use, land_type, ground_condition, ground_relative_permittivity, relative_groundwater_level)
    VALUES
        (uuid_generate_v4(), twente_project_id, 'Twente Site 01', 'Construction Site A, Enschede, Netherlands',
         ST_SetSRID(ST_Point(6.8936, 52.2387), 4326), 'Construction', 'Street', 'Sandy', 8.16, 'Medium'),
        (uuid_generate_v4(), twente_project_id, 'Twente Site 02', 'Industrial Area B, Enschede, Netherlands',
         ST_SetSRID(ST_Point(6.8956, 52.2407), 4326), 'Industrial', 'Sidewalk', 'Clayey', 19.46, 'High'),
        (uuid_generate_v4(), twente_project_id, 'Twente Site 03', 'Residential Area C, Enschede, Netherlands',
         ST_SetSRID(ST_Point(6.8976, 52.2427), 4326), 'Residential', 'Greenery', 'Mixed', 12.34, 'Low'),
        (uuid_generate_v4(), twente_project_id, 'Twente Site 04', 'Commercial District D, Enschede, Netherlands',
         ST_SetSRID(ST_Point(6.8996, 52.2447), 4326), 'Commercial', 'Street', 'Sandy', 9.87, 'Medium')
    ON CONFLICT DO NOTHING;

    RAISE NOTICE 'Twente survey sites created successfully';
END $$;

-- Insert comprehensive GPR survey data with all 25+ Twente metadata fields
DO $$
DECLARE
    site_record RECORD;
    survey_id UUID;
BEGIN
    FOR site_record IN SELECT * FROM survey_sites WHERE site_name LIKE 'Twente Site%' LOOP
        -- Create detailed survey with comprehensive metadata
        INSERT INTO gpr_surveys (
            id, project_id, site_id, location_id,
            utility_surveying_objective, construction_workers, exact_location_accuracy_required,
            complementary_works, land_cover, terrain_levelling, terrain_smoothness, weather_condition,
            rubble_presence, tree_roots_presence, polluted_soil_presence, blast_furnace_slag_presence,
            amount_of_utilities, utility_crossing, utility_path_linear,
            survey_date, surveyor_name, equipment_model, antenna_frequency, trace_spacing,
            samples_per_trace, time_range_ns, quality_level, confidence_score
        ) VALUES (
            uuid_generate_v4(), site_record.project_id, site_record.id,
            CASE
                WHEN site_record.site_name = 'Twente Site 01' THEN '01.1'
                WHEN site_record.site_name = 'Twente Site 02' THEN '01.2'
                WHEN site_record.site_name = 'Twente Site 03' THEN '02.1'
                ELSE '02.2'
            END,
            'Utility detection for construction planning', 'Yes', TRUE,
            ARRAY['excavation_planning', 'service_connections'],
            CASE
                WHEN site_record.land_type = 'Street' THEN 'Brick road concrete'
                WHEN site_record.land_type = 'Sidewalk' THEN 'Concrete pavement'
                ELSE 'Grass/vegetation'
            END,
            'Flat', 'Smooth', 'Dry',
            CASE WHEN site_record.site_name IN ('Twente Site 02', 'Twente Site 04') THEN TRUE ELSE FALSE END,
            CASE WHEN site_record.land_type = 'Greenery' THEN TRUE ELSE FALSE END,
            FALSE, FALSE,
            CASE
                WHEN site_record.site_name = 'Twente Site 01' THEN 3
                WHEN site_record.site_name = 'Twente Site 02' THEN 5
                WHEN site_record.site_name = 'Twente Site 03' THEN 2
                ELSE 4
            END,
            CASE WHEN site_record.site_name IN ('Twente Site 02', 'Twente Site 04') THEN TRUE ELSE FALSE END,
            TRUE,
            CURRENT_DATE - INTERVAL '6 months' + (EXTRACT(DOY FROM NOW()) || ' days')::INTERVAL,
            'Dr. Survey Specialist', 'GSSI 500MHz', 500, 0.02, 512, 50, 'QL-B', 0.87
        ) RETURNING id INTO survey_id;

        RAISE NOTICE 'Created survey for site: %', site_record.site_name;
    END LOOP;
END $$;

-- Insert environmental metadata for Twente surveys
DO $$
DECLARE
    survey_record RECORD;
    ground_condition_id INTEGER;
    weather_condition_id INTEGER;
    land_cover_id INTEGER;
    land_use_id INTEGER;
BEGIN
    -- Get lookup table IDs
    SELECT id INTO ground_condition_id FROM ground_conditions WHERE condition_code = 'SAND' LIMIT 1;
    SELECT id INTO weather_condition_id FROM weather_conditions WHERE condition_code = 'DRY' LIMIT 1;
    SELECT id INTO land_cover_id FROM surface_materials WHERE material_code = 'CONC' LIMIT 1;
    SELECT id INTO land_use_id FROM land_use_types WHERE land_use_code = 'CONSTR' LIMIT 1;

    FOR survey_record IN SELECT * FROM gpr_surveys WHERE location_id LIKE '0%' LOOP
        INSERT INTO environmental_metadata (
            id, survey_id, location_id, site_classification, surveying_objective,
            construction_workers_present, exact_location_accuracy_required,
            ground_condition_id, ground_relative_permittivity, relative_groundwater_level,
            land_cover_id, terrain_levelling, terrain_smoothness, weather_condition_id,
            temperature_celsius, humidity_percentage,
            rubble_presence, tree_roots_presence, polluted_soil_presence, blast_furnace_slag_presence,
            amount_of_utilities, utility_crossing, utility_path_linear,
            metadata_completeness_score, data_reliability_score, collected_by
        ) VALUES (
            uuid_generate_v4(), survey_record.id, survey_record.location_id,
            'Urban construction site', survey_record.utility_surveying_objective,
            'Yes', survey_record.exact_location_accuracy_required,
            ground_condition_id,
            CASE
                WHEN survey_record.location_id = '01.1' THEN 8.16
                WHEN survey_record.location_id = '01.2' THEN 19.46
                WHEN survey_record.location_id = '02.1' THEN 12.34
                ELSE 9.87
            END,
            CASE
                WHEN survey_record.location_id IN ('01.1', '02.2') THEN 'Medium'
                WHEN survey_record.location_id = '01.2' THEN 'High'
                ELSE 'Low'
            END,
            land_cover_id, 'Flat', 'Smooth', weather_condition_id,
            18.5, 65.0,
            survey_record.rubble_presence, survey_record.tree_roots_presence,
            survey_record.polluted_soil_presence, survey_record.blast_furnace_slag_presence,
            survey_record.amount_of_utilities, survey_record.utility_crossing, survey_record.utility_path_linear,
            0.95, 0.90, 'Twente Research Team'
        );
    END LOOP;

    RAISE NOTICE 'Environmental metadata created for Twente surveys';
END $$;

-- ================================================================
-- MOJAHID GPR IMAGES DATASET SAMPLE DATA
-- 2,239+ images with 6 categories
-- ================================================================

-- Insert sample GPR image data representing the Mojahid dataset
DO $$
DECLARE
    survey_record RECORD;
    image_categories TEXT[] := ARRAY['cavities', 'intact', 'utilities', 'augmented_cavities', 'augmented_intact', 'augmented_utilities'];
    category TEXT;
    i INTEGER;
BEGIN
    -- Create sample images for each survey
    FOR survey_record IN SELECT * FROM gpr_surveys WHERE location_id LIKE '0%' LOOP
        FOREACH category IN ARRAY image_categories LOOP
            FOR i IN 1..3 LOOP -- 3 images per category per survey
                INSERT INTO gpr_image_data (
                    id, survey_id, image_filename, image_path, image_category, image_type,
                    width_pixels, height_pixels, bit_depth, color_space,
                    predicted_class, prediction_confidence, feature_vector,
                    ground_truth_class, manually_verified, verified_by
                ) VALUES (
                    uuid_generate_v4(), survey_record.id,
                    format('%s_%s_%s.png', survey_record.location_id, category, i),
                    format('/data/mojahid/%s/%s_%s_%s.png', category, survey_record.location_id, category, i),
                    category,
                    CASE WHEN category LIKE 'augmented_%' THEN 'augmented' ELSE 'original' END,
                    512, 256, 8, 'grayscale',
                    category, 0.85 + (RANDOM() * 0.14), -- Confidence between 0.85-0.99
                    ARRAY(SELECT RANDOM() FROM generate_series(1, 512))::vector(512),
                    category, TRUE, 'Mojahid Dataset Team'
                );
            END LOOP;
        END LOOP;
    END LOOP;

    RAISE NOTICE 'Sample Mojahid GPR images created';
END $$;

-- Add image annotations for object detection
DO $$
DECLARE
    image_record RECORD;
BEGIN
    FOR image_record IN SELECT * FROM gpr_image_data WHERE image_category IN ('utilities', 'augmented_utilities') LOOP
        -- Add bounding box annotations for utility objects
        INSERT INTO image_annotations (
            id, image_id, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max,
            annotation_class, confidence_score, annotator, annotation_method
        ) VALUES (
            uuid_generate_v4(), image_record.id,
            0.2 + (RANDOM() * 0.2), 0.3 + (RANDOM() * 0.2), -- Random bbox starting position
            0.6 + (RANDOM() * 0.2), 0.7 + (RANDOM() * 0.2), -- Random bbox ending position
            'utility_pipe', 0.92, 'Expert Annotator', 'manual'
        );
    END LOOP;

    RAISE NOTICE 'Image annotations created';
END $$;

-- ================================================================
-- DETECTED UTILITIES SAMPLE DATA
-- Realistic utility detection results
-- ================================================================

DO $$
DECLARE
    survey_record RECORD;
    utility_materials TEXT[] := ARRAY['steel', 'polyVinylChloride', 'asbestosCement', 'concrete', 'ductileIron'];
    utility_disciplines TEXT[] := ARRAY['electricity', 'water', 'sewer', 'telecommunications', 'gas'];
    i INTEGER;
BEGIN
    FOR survey_record IN SELECT * FROM gpr_surveys WHERE location_id LIKE '0%' LOOP
        -- Create detected utilities based on the amount_of_utilities field
        FOR i IN 1..survey_record.amount_of_utilities LOOP
            INSERT INTO detected_utilities (
                id, survey_id, detection_line, position_along_line_m, depth_m,
                coordinates, utility_discipline, utility_material, utility_diameter_mm,
                additional_info, detection_method, confidence_score, false_positive_probability
            ) VALUES (
                uuid_generate_v4(), survey_record.id,
                format('Line_%s', i),
                5.0 + (i * 3.5), -- Spread utilities along survey line
                0.8 + (RANDOM() * 1.5), -- Depth between 0.8-2.3m
                ST_SetSRID(ST_Point(
                    ST_X((SELECT location FROM survey_sites WHERE id = survey_record.site_id)) + (RANDOM() * 0.001),
                    ST_Y((SELECT location FROM survey_sites WHERE id = survey_record.site_id)) + (RANDOM() * 0.001)
                ), 4326),
                utility_disciplines[1 + (i-1) % array_length(utility_disciplines, 1)],
                utility_materials[1 + (i-1) % array_length(utility_materials, 1)],
                CASE
                    WHEN utility_disciplines[1 + (i-1) % array_length(utility_disciplines, 1)] = 'electricity' THEN 100 + RANDOM() * 50
                    WHEN utility_disciplines[1 + (i-1) % array_length(utility_disciplines, 1)] = 'water' THEN 200 + RANDOM() * 400
                    ELSE 150 + RANDOM() * 200
                END,
                CASE WHEN i % 3 = 0 THEN 'bundled_with_other_utilities' ELSE 'single_utility' END,
                'hyperbola_detection', 0.75 + (RANDOM() * 0.2), 0.05 + (RANDOM() * 0.1)
            );
        END LOOP;
    END LOOP;

    RAISE NOTICE 'Detected utilities created';
END $$;

-- ================================================================
-- GROUND TRUTH VALIDATION SAMPLE DATA
-- Accuracy assessment and validation campaigns
-- ================================================================

-- Create validation campaigns
DO $$
DECLARE
    twente_project_id UUID;
    campaign_id UUID;
BEGIN
    SELECT id INTO twente_project_id FROM projects WHERE name = 'Twente University GPR Research Dataset';

    INSERT INTO validation_campaigns (
        id, project_id, campaign_name, primary_objective, target_accuracy_mm,
        planned_start_date, planned_end_date, actual_start_date, actual_end_date,
        campaign_manager, responsible_engineer, validation_standard,
        total_planned_validations, total_completed_validations,
        overall_accuracy_percentage, precision_percentage, recall_percentage, f1_score,
        status
    ) VALUES (
        uuid_generate_v4(), twente_project_id, 'Twente Validation Campaign 2023',
        'Validate GPR detection accuracy against excavated ground truth',
        200, '2023-06-01', '2023-08-31', '2023-06-05', '2023-08-28',
        'Validation Manager', 'Lead Validation Engineer', 'PAS128:2022',
        50, 47, 82.5, 85.3, 79.8, 0.824, 'completed'
    ) RETURNING id INTO campaign_id;

    -- Create individual validation records
    DECLARE
        detected_utility_record RECORD;
        validation_methods INTEGER[] := ARRAY[1, 2, 3]; -- Trial trench, vacuum excavation, hand digging
    BEGIN
        FOR detected_utility_record IN
            SELECT du.*, s.id as survey_id FROM detected_utilities du
            JOIN gpr_surveys s ON du.survey_id = s.id
            WHERE s.location_id LIKE '0%'
            LIMIT 15 -- Sample of detections to validate
        LOOP
            INSERT INTO ground_truth_validations (
                id, campaign_id, survey_id, detected_utility_id,
                validation_point, validation_method_id, validation_depth_m,
                validation_date, validator_name,
                utility_present, actual_utility_type, actual_material, actual_diameter_mm, actual_depth_m,
                horizontal_position_error_mm, depth_error_mm,
                detection_result, validation_confidence,
                material_identification_correct, field_notes
            ) VALUES (
                uuid_generate_v4(), campaign_id, detected_utility_record.survey_id, detected_utility_record.id,
                detected_utility_record.coordinates,
                validation_methods[1 + (RANDOM() * (array_length(validation_methods, 1) - 1))::INTEGER],
                detected_utility_record.depth_m + (RANDOM() * 0.3), -- Slight variation in excavation depth
                CURRENT_DATE - INTERVAL '60 days' + (RANDOM() * INTERVAL '30 days'),
                'Field Validation Engineer',
                TRUE, -- Utility confirmed present
                detected_utility_record.utility_discipline,
                detected_utility_record.utility_material,
                detected_utility_record.utility_diameter_mm + (-20 + RANDOM() * 40)::INTEGER, -- Slight diameter variation
                detected_utility_record.depth_m + (-0.15 + RANDOM() * 0.3), -- Actual depth with some variation
                (-100 + RANDOM() * 200)::INTEGER, -- Horizontal error ±100mm
                (-50 + RANDOM() * 100)::INTEGER, -- Depth error ±50mm
                CASE
                    WHEN RANDOM() > 0.15 THEN 'true_positive'::detection_result_enum
                    ELSE 'false_positive'::detection_result_enum
                END,
                CASE
                    WHEN RANDOM() > 0.2 THEN 'high'::confidence_level_enum
                    ELSE 'medium'::confidence_level_enum
                END,
                RANDOM() > 0.25, -- 75% material identification accuracy
                'Standard excavation validation. Weather conditions: clear. Ground conditions: as expected.'
            );
        END LOOP;
    END;

    RAISE NOTICE 'Ground truth validation data created';
END $$;

-- ================================================================
-- MACHINE LEARNING MODEL SAMPLE DATA
-- Model performance tracking and cross-validation
-- ================================================================

-- Insert sample ML models
DO $$
DECLARE
    model_id UUID;
    cv_experiment_id UUID;
BEGIN
    -- Classification model for GPR image analysis
    INSERT INTO ml_models (
        id, model_name, model_version, category_id, architecture_id,
        primary_objective, target_datasets, input_specifications, output_specifications,
        total_parameters, trainable_parameters, model_size_mb,
        training_dataset_size, validation_dataset_size, test_dataset_size,
        training_duration_hours, development_stage, created_by, model_description
    ) VALUES (
        uuid_generate_v4(), 'GPR-UtilityNet-v1', '1.0.0', 1, 1, -- Classification, ResNet50
        'Classify GPR images into utility categories',
        ARRAY['Mojahid', 'Twente'],
        '{"input_shape": [224, 224, 3], "preprocessing": "normalize", "augmentation": true}'::JSONB,
        '{"num_classes": 6, "output_activation": "softmax"}'::JSONB,
        25600000, 25600000, 98.5,
        1800, 450, 225, 24.5, 'production', 'ML Research Team',
        'Deep learning model for automated GPR image classification using transfer learning from ResNet50'
    ) RETURNING id INTO model_id;

    -- Cross-validation experiment
    INSERT INTO cross_validation_experiments (
        id, model_id, experiment_name, cv_strategy, n_folds,
        dataset_identifier, total_samples, experiment_start, experiment_end,
        mean_accuracy, std_accuracy, mean_precision, std_precision, mean_recall, std_recall,
        mean_f1_score, std_f1_score, mean_auc_roc, std_auc_roc,
        performance_stability_score, total_training_time_minutes, status
    ) VALUES (
        uuid_generate_v4(), model_id, 'GPR-UtilityNet 5-fold CV', 'stratified_k_fold', 5,
        'Mojahid_Combined_Dataset', 2239,
        NOW() - INTERVAL '5 days', NOW() - INTERVAL '4 days',
        0.8743, 0.0234, 0.8821, 0.0189, 0.8654, 0.0267,
        0.8734, 0.0198, 0.9432, 0.0156,
        0.9801, 1473.5, 'completed'
    ) RETURNING id INTO cv_experiment_id;

    -- Individual fold results
    DECLARE
        fold_num INTEGER;
        base_accuracy DECIMAL := 0.8743;
    BEGIN
        FOR fold_num IN 1..5 LOOP
            INSERT INTO cv_fold_results (
                id, cv_experiment_id, fold_number, train_size, validation_size,
                training_duration_minutes, training_epochs,
                validation_accuracy, validation_precision, validation_recall, validation_f1_score, validation_auc_roc,
                confusion_matrix, convergence_achieved
            ) VALUES (
                uuid_generate_v4(), cv_experiment_id, fold_num - 1, 1791, 448,
                294.7, 25,
                base_accuracy + (-0.03 + RANDOM() * 0.06), -- Variation around mean
                0.8821 + (-0.02 + RANDOM() * 0.04),
                0.8654 + (-0.03 + RANDOM() * 0.06),
                0.8734 + (-0.025 + RANDOM() * 0.05),
                0.9432 + (-0.02 + RANDOM() * 0.04),
                format('{"0": {"0": %s, "1": %s}, "1": {"0": %s, "1": %s}}',
                       85 + (RANDOM() * 10)::INTEGER, 5 + (RANDOM() * 3)::INTEGER,
                       7 + (RANDOM() * 3)::INTEGER, 83 + (RANDOM() * 10)::INTEGER)::JSONB,
                TRUE
            );
        END LOOP;
    END;

    RAISE NOTICE 'ML model and cross-validation data created';
END $$;

-- ================================================================
-- PAS 128 COMPLIANCE SAMPLE DATA
-- Quality level assessments and compliance tracking
-- ================================================================

DO $$
DECLARE
    survey_record RECORD;
    target_ql_id INTEGER;
BEGIN
    SELECT id INTO target_ql_id FROM pas128_quality_levels WHERE quality_level_code = 'QL-B';

    FOR survey_record IN SELECT * FROM gpr_surveys WHERE location_id LIKE '0%' LOOP
        INSERT INTO pas128_compliance_assessments (
            id, survey_id, assessment_reference, assessor_name, assessor_certification,
            target_quality_level_id, achieved_quality_level_id,
            detection_methods_used, primary_method_id,
            horizontal_accuracy_achieved_mm, vertical_accuracy_achieved_mm,
            sample_size_total, sample_size_verified, verification_percentage,
            survey_method_documented, coordinate_system_specified, limitations_documented,
            confidence_levels_assigned, equipment_calibrated, operator_certified,
            compliant, compliance_score, compliance_grade,
            approval_status
        ) VALUES (
            uuid_generate_v4(), survey_record.id,
            format('PAS128-ASSESS-%s-%s', survey_record.location_id, extract(year from CURRENT_DATE)),
            'PAS 128 Compliance Assessor', 'Certified PAS 128 Assessor Level 2',
            target_ql_id, target_ql_id,
            ARRAY[5, 6], 5, -- GPR methods
            150 + (RANDOM() * 100)::INTEGER, 250 + (RANDOM() * 100)::INTEGER,
            survey_record.amount_of_utilities,
            CASE WHEN survey_record.amount_of_utilities > 0 THEN (survey_record.amount_of_utilities * 0.6)::INTEGER ELSE 0 END,
            CASE WHEN survey_record.amount_of_utilities > 0 THEN 60.0 + (RANDOM() * 30) ELSE 0 END,
            TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
            RANDOM() > 0.25, -- 75% compliance rate
            75.0 + (RANDOM() * 20),
            CASE
                WHEN RANDOM() > 0.7 THEN 'Excellent'
                WHEN RANDOM() > 0.4 THEN 'Good'
                ELSE 'Adequate'
            END,
            'approved'
        );
    END LOOP;

    RAISE NOTICE 'PAS 128 compliance assessments created';
END $$;

-- ================================================================
-- USAG STRIKE REPORTS SAMPLE DATA
-- Historical incident data for pattern analysis
-- ================================================================

DO $$
DECLARE
    incident_categories INTEGER[] := ARRAY[1, 2, 3, 4, 5]; -- Different incident types
    utility_types INTEGER[] := ARRAY[1, 2, 3, 4, 5]; -- Different utility types
    i INTEGER;
    base_location GEOMETRY := ST_SetSRID(ST_Point(6.8936, 52.2387), 4326); -- Near Twente area
BEGIN
    FOR i IN 1..25 LOOP -- Create 25 sample incidents
        INSERT INTO usag_strike_incidents (
            id, incident_reference, incident_category_id, utility_type_id,
            incident_date, incident_time, report_date,
            incident_location, location_description, address,
            site_type, excavation_type, excavation_method,
            incident_description, immediate_cause,
            utility_survey_conducted, survey_method, survey_quality_level,
            utility_damaged, damage_extent, service_interruption, customers_affected,
            duration_of_outage_hours, safety_incident, estimated_repair_cost,
            excavation_contractor, client_organization, investigation_status,
            data_source, data_quality_score, data_completeness_score
        ) VALUES (
            uuid_generate_v4(),
            format('USAG-2023-%04d', 1000 + i),
            incident_categories[1 + (i-1) % array_length(incident_categories, 1)],
            utility_types[1 + (i-1) % array_length(utility_types, 1)],
            CURRENT_DATE - INTERVAL '1 year' + (RANDOM() * INTERVAL '365 days'),
            TIME '08:00:00' + (RANDOM() * INTERVAL '10 hours'),
            CURRENT_DATE - INTERVAL '1 year' + (RANDOM() * INTERVAL '365 days') + INTERVAL '1 day',
            ST_SetSRID(ST_Point(
                ST_X(base_location) + (-0.05 + RANDOM() * 0.1),
                ST_Y(base_location) + (-0.05 + RANDOM() * 0.1)
            ), 4326),
            format('Construction site incident %s', i),
            format('Industrial Area, Location %s, Netherlands', chr(64 + i)),
            CASE (i % 4)
                WHEN 0 THEN 'residential'
                WHEN 1 THEN 'commercial'
                WHEN 2 THEN 'industrial'
                ELSE 'highway'
            END,
            CASE (i % 3)
                WHEN 0 THEN 'planned_excavation'
                WHEN 1 THEN 'emergency_repair'
                ELSE 'maintenance'
            END,
            CASE (i % 3)
                WHEN 0 THEN 'mechanical_excavator'
                WHEN 1 THEN 'hand_digging'
                ELSE 'drilling'
            END,
            format('Utility strike incident during excavation work. Case %s details.', i),
            CASE (i % 4)
                WHEN 0 THEN 'inadequate_location'
                WHEN 1 THEN 'excavation_error'
                WHEN 2 THEN 'equipment_failure'
                ELSE 'procedural_violation'
            END,
            RANDOM() > 0.3, -- 70% had surveys
            CASE WHEN RANDOM() > 0.3 THEN 'GPR' ELSE 'CAT_scanner' END,
            CASE WHEN RANDOM() > 0.3 THEN 'QL-B' ELSE 'QL-C' END,
            TRUE,
            CASE (i % 3)
                WHEN 0 THEN 'minor'
                WHEN 1 THEN 'moderate'
                ELSE 'severe'
            END,
            RANDOM() > 0.4, -- 60% caused service interruption
            CASE WHEN RANDOM() > 0.4 THEN (10 + RANDOM() * 1000)::INTEGER ELSE 0 END,
            CASE WHEN RANDOM() > 0.4 THEN (1 + RANDOM() * 24) ELSE 0 END,
            RANDOM() > 0.8, -- 20% safety incidents
            1000 + (RANDOM() * 50000), -- Repair costs between £1k-£50k
            format('Contractor-%s', chr(65 + (i % 10))),
            format('Client-Organization-%s', chr(65 + (i % 5))),
            CASE (i % 3)
                WHEN 0 THEN 'completed'
                WHEN 1 THEN 'ongoing'
                ELSE 'pending'
            END,
            'usag_report', 0.85 + (RANDOM() * 0.14), 0.80 + (RANDOM() * 0.19)
        );
    END LOOP;

    RAISE NOTICE 'USAG strike incident data created';
END $$;

-- ================================================================
-- SIGNAL PROCESSING SAMPLE DATA
-- Enhanced signal analysis and feature extraction
-- ================================================================

DO $$
DECLARE
    survey_record RECORD;
    signal_id UUID;
    trace_num INTEGER;
BEGIN
    -- Create signal timeseries data for surveys
    FOR survey_record IN SELECT * FROM gpr_surveys WHERE location_id LIKE '0%' LOOP
        FOR trace_num IN 1..20 LOOP -- 20 traces per survey
            INSERT INTO gpr_signal_timeseries (
                id, survey_id, trace_number, total_traces, sample_rate_mhz, time_window_ns,
                signal_envelope, instantaneous_phase, instantaneous_frequency,
                signal_energy_distribution, two_way_travel_time_ns, estimated_dielectric_constant,
                signal_to_clutter_ratio, coherence_coefficient, signal_stability_index,
                processing_algorithm, processing_parameters
            ) VALUES (
                uuid_generate_v4(), survey_record.id, trace_num, 100, 500.0, 50.0,
                ARRAY(SELECT RANDOM() FROM generate_series(1, 512))::vector(512),
                ARRAY(SELECT RANDOM() * 2 * PI() FROM generate_series(1, 512))::vector(512),
                ARRAY(SELECT 400 + RANDOM() * 200 FROM generate_series(1, 512))::vector(512),
                ARRAY(SELECT RANDOM() FROM generate_series(1, 64))::vector(64),
                8.5 + (RANDOM() * 15), 8.0 + (RANDOM() * 4),
                12.5 + (RANDOM() * 8), 0.75 + (RANDOM() * 0.2), 0.85 + (RANDOM() * 0.14),
                'advanced_signal_processor', '{"filter_type": "bandpass", "freq_range": [400, 600]}'::JSONB
            ) RETURNING id INTO signal_id;

            -- Add frequency analysis for some signals
            IF trace_num % 5 = 0 THEN
                INSERT INTO gpr_frequency_analysis (
                    id, signal_timeseries_id, frequency_spectrum, magnitude_spectrum,
                    peak_frequency_mhz, bandwidth_3db_mhz, spectral_centroid_mhz,
                    wavelet_coefficients, fft_size
                ) VALUES (
                    uuid_generate_v4(), signal_id,
                    ARRAY(SELECT RANDOM() FROM generate_series(1, 256))::vector(256),
                    ARRAY(SELECT RANDOM() FROM generate_series(1, 256))::vector(256),
                    480 + (RANDOM() * 40), 45 + (RANDOM() * 20), 495 + (RANDOM() * 30),
                    ARRAY(SELECT RANDOM() FROM generate_series(1, 512))::vector(512),
                    512
                );
            END IF;
        END LOOP;
    END LOOP;

    RAISE NOTICE 'Signal processing data created';
END $$;

-- ================================================================
-- DATA VALIDATION AND STATISTICS
-- ================================================================

-- Refresh all materialized views
SELECT refresh_all_materialized_views();

-- Update campaign statistics
DO $$
DECLARE
    campaign_record RECORD;
BEGIN
    FOR campaign_record IN SELECT id FROM validation_campaigns LOOP
        PERFORM update_campaign_statistics(campaign_record.id);
    END LOOP;

    RAISE NOTICE 'Campaign statistics updated';
END $$;

-- Final data validation and statistics
DO $$
DECLARE
    table_stats RECORD;
    total_records INTEGER := 0;
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '================================================================';
    RAISE NOTICE 'SAMPLE DATA LOADING COMPLETED';
    RAISE NOTICE '================================================================';
    RAISE NOTICE '';

    -- Show record counts for key tables
    FOR table_stats IN
        SELECT
            schemaname,
            tablename,
            n_tup_ins AS row_count
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
        AND tablename IN (
            'projects', 'survey_sites', 'gpr_surveys', 'environmental_metadata',
            'detected_utilities', 'gpr_image_data', 'ground_truth_validations',
            'ml_models', 'cross_validation_experiments', 'pas128_compliance_assessments',
            'usag_strike_incidents', 'gpr_signal_timeseries'
        )
        ORDER BY n_tup_ins DESC
    LOOP
        RAISE NOTICE '% records in %', table_stats.row_count, table_stats.tablename;
        total_records := total_records + table_stats.row_count;
    END LOOP;

    RAISE NOTICE '';
    RAISE NOTICE 'Total sample records created: %', total_records;
    RAISE NOTICE '';
    RAISE NOTICE 'DATASET SUMMARY:';
    RAISE NOTICE '• Twente GPR Dataset: Survey data with 25+ environmental metadata fields';
    RAISE NOTICE '• Mojahid Images: GPR image classification data (6 categories)';
    RAISE NOTICE '• Ground Truth: Validation campaigns with accuracy assessment';
    RAISE NOTICE '• ML Models: Cross-validation experiments and performance tracking';
    RAISE NOTICE '• PAS 128: Compliance assessments and quality level determination';
    RAISE NOTICE '• USAG Reports: Historical utility strike incident data';
    RAISE NOTICE '• Signal Processing: Advanced GPR signal analysis features';
    RAISE NOTICE '';
    RAISE NOTICE 'Ready for: Analysis, ML training, Compliance reporting, Pattern detection';
    RAISE NOTICE '================================================================';
END $$;