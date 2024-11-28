import os
from main import query_translate_and_assess

def load_or_optimize_threshold(
    acceptance_threshold_file,
    model_type_T,
    model_type_C,
    combined_df,
    llama3_api_endpoint,
    openai_api_key,
    initial_T_A=0.1,
    learning_rate=0.01,
    num_iterations=50,
    k=3
):
    """Load the best threshold from file or optimize it using gradient descent."""
    if os.path.exists(acceptance_threshold_file):
        with open(acceptance_threshold_file, "r") as file:
            best_T_A = float(file.read())
    else:
        # Determine threshold using gradient descent
        best_T_A = gradient_descent_threshold_optimization(
            model_type_T,
            model_type_C,
            combined_df,
            llama3_api_endpoint,
            openai_api_key,
            initial_T_A=initial_T_A,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            k=k,
        )
        with open(acceptance_threshold_file, "w") as file:
            file.write(f"{best_T_A}")

    print(f"Optimized Acceptance Threshold (T_A): {best_T_A}")

    return best_T_A


def gradient_descent_threshold_optimization(
    model_type_T,
    model_type_C,
    combined_df,
    llama3_api_endpoint,
    openai_api_key,
    initial_T_A=0.1,
    learning_rate=0.01,
    num_iterations=50,
    k=3,
):
    T_A = initial_T_A

    for iteration in range(num_iterations):
        total_quality_score = 0
        total_samples = 0

        for _ , row in combined_df.iterrows():
            Q = row["sparql_wikidata"]
            _, _, quality_score = query_translate_and_assess(
                model_type_T,
                Q,
                T_A,
                model_type_C,
                llama3_api_endpoint=llama3_api_endpoint,
                openai_api_key=openai_api_key,
                k=k,
            )
            total_quality_score += quality_score
            total_samples += 1

        avg_quality_score = total_quality_score / total_samples

        # Update thresholds
        T_A_gradient = -1 if avg_quality_score < T_A else 1

        T_A = max(0, min(1, T_A + learning_rate * T_A_gradient))

        print(
            f"Iteration {iteration + 1}: T_A = {T_A:.4f}, Avg Quality Score = {avg_quality_score:.4f}"
        )

    return T_A