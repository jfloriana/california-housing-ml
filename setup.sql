-- ============================================================
-- California Housing ML - Supabase Setup
-- ============================================================

-- 1. Users table (managed by Supabase Auth, but we store preferences)
CREATE TABLE IF NOT EXISTS public.user_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE UNIQUE,
    language TEXT DEFAULT 'es' CHECK (language IN ('es', 'en')),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE public.user_preferences ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own preferences"
    ON public.user_preferences FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own preferences"
    ON public.user_preferences FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own preferences"
    ON public.user_preferences FOR UPDATE
    USING (auth.uid() = user_id);

-- 2. Training metrics storage
CREATE TABLE IF NOT EXISTS public.training_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    mse FLOAT,
    rmse FLOAT,
    mae FLOAT,
    r2 FLOAT,
    training_time_sec FLOAT,
    num_params INT,
    created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE public.training_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can read training metrics"
    ON public.training_metrics FOR SELECT
    USING (true);

CREATE POLICY "Service role can insert training metrics"
    ON public.training_metrics FOR INSERT
    WITH CHECK (true);

-- 3. Predictions log
CREATE TABLE IF NOT EXISTS public.predictions_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    input_data JSONB,
    prediction FLOAT,
    model_used TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE public.predictions_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own predictions"
    ON public.predictions_log FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Service role can insert predictions"
    ON public.predictions_log FOR INSERT
    WITH CHECK (true);

-- 4. Hyperparameter tuning results
CREATE TABLE IF NOT EXISTS public.hyperparameter_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT NOT NULL,
    params JSONB,
    best_mse FLOAT,
    best_r2 FLOAT,
    created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE public.hyperparameter_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can read hyperparameter results"
    ON public.hyperparameter_results FOR SELECT
    USING (true);

CREATE POLICY "Service role can insert hyperparameter results"
    ON public.hyperparameter_results FOR INSERT
    WITH CHECK (true);

-- 5. Cross validation results
CREATE TABLE IF NOT EXISTS public.cross_validation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT NOT NULL,
    n_folds INT,
    fold_scores JSONB,
    mean_score FLOAT,
    std_score FLOAT,
    created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE public.cross_validation_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can read cv results"
    ON public.cross_validation_results FOR SELECT
    USING (true);

CREATE POLICY "Service role can insert cv results"
    ON public.cross_validation_results FOR INSERT
    WITH CHECK (true);

-- 6. Storage bucket for model files
-- Run this in the SQL editor:
-- INSERT INTO storage.buckets (id, name, public) VALUES ('models', 'models', true);
-- INSERT INTO storage.buckets (id, name, public) VALUES ('reports', 'reports', true);
