def gpt4turbo_cost(total_instances):
    # Cost calculation for running a generative model on a dataset

    # Cost rates
    cost_per_1k_input_tokens = 0.01  # in dollars
    cost_per_1k_output_tokens = 0.03  # in dollars

    # Maximum tokens per instance
    max_input_tokens_per_instance = 512
    max_output_tokens_per_instance = 512


    # Calculating total tokens for all instances
    total_input_tokens = max_input_tokens_per_instance * total_instances
    total_output_tokens = max_output_tokens_per_instance * total_instances

    # Calculating costs
    total_input_cost = (total_input_tokens / 1000) * cost_per_1k_input_tokens
    total_output_cost = (total_output_tokens / 1000) * cost_per_1k_output_tokens

    # Total cost
    total_cost = total_input_cost + total_output_cost
    return total_cost

if __name__ == "__main__":
    # Total number of instances
    total_instances = 10000
    total_cost = gpt4turbo_cost(total_instances)

    print(total_cost * 10 * 20)