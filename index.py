import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sympy import symbols, Eq, solve

# Initialize session state
if "y_euler" not in st.session_state:
    st.session_state.y_euler = None
    st.session_state.y_improved = None
    st.session_state.x_euler = None
    st.session_state.x_improved = None
# Function to sanitize and parse the equation
def sanitize_and_format_equation(raw_eq):
    # Replace `y'` with a placeholder that indicates derivative handling
    if "y'" in raw_eq:
        raw_eq = raw_eq.replace("y'", "")
    sanitized = raw_eq.replace("^", "**").replace("e^", "np.exp(")
    
    # Ensure multiplication is explicitly stated
    sanitized = "".join(
        f"*{char}" if char.isalpha() and i > 0 and sanitized[i-1].isdigit() else char
        for i, char in enumerate(sanitized)
    )
    return sanitized


# Function to format the equation for Python compatibility
def format_equation(equation):
    formatted_eq = equation.replace("^", "**").replace("e^", "np.exp(")
    formatted_eq = "".join(f"*{char}" if char.isalpha() and i > 0 and formatted_eq[i-1].isdigit() else char for i, char in enumerate(formatted_eq))
    return formatted_eq

# Euler Method implementation
def euler_method(differential_eq, x0, y0, h, steps):
    x_values = [x0]
    y_values = [y0]
    for i in range(steps):
        try:
            # Evaluate f(x, y) for y'
            f_xy = eval(differential_eq, {"x": x_values[-1], "y": y_values[-1], "np": np})
            y_next = y_values[-1] + h * f_xy
            x_next = x_values[-1] + h
        except Exception as e:
            raise ValueError(f"Error solving Euler method at step {i}: {e}")
        x_values.append(x_next)
        y_values.append(y_next)
    return x_values, y_values

# Improved Euler Method implementation
def improved_euler_method(differential_eq, x0, y0, h, steps):
    x_values = [x0]
    y_values = [y0]
    for i in range(steps):
        try:
            k1 = eval(differential_eq, {"x": x_values[-1], "y": y_values[-1], "np": np})
            k2 = eval(differential_eq, {"x": x_values[-1] + h, "y": y_values[-1] + h * k1, "np": np})
        except Exception as e:
            raise ValueError(f"Error solving Improved Euler method at step {i}: {e}")
        y_next = y_values[-1] + (h / 2) * (k1 + k2)
        x_next = x_values[-1] + h
        x_values.append(x_next)
        y_values.append(y_next)
    return x_values, y_values

# Function for Simpson's 1/3rd Rule
# Simpson's 1/3 Rule
# Simpson's 1/3 Rule Implementation
def simpsons_one_third_rule(points):
    n = len(points) - 1
    h = (points[-1][0] - points[0][0]) / n
    result = points[0][1] + points[-1][1]

    for i in range(1, n):
        weight = 4 if i % 2 != 0 else 2
        result += weight * points[i][1]

    result *= h / 3
    return result, "Simpson's 1/3 Rule"


# Simpson's 3/8 Rule Implementation
def simpsons_three_eighths_rule(points):
    n = len(points) - 1
    h = (points[-1][0] - points[0][0]) / n
    result = points[0][1] + points[-1][1]

    for i in range(1, n):
        weight = 3 if i % 3 != 0 else 2
        result += weight * points[i][1]

    result *= 3 * h / 8
    return result, "Simpson's 3/8 Rule"

def error_analysis(true_values, euler_values, improved_euler_values):
    euler_error = np.abs(np.array(true_values) - np.array(euler_values))
    improved_error = np.abs(np.array(true_values) - np.array(improved_euler_values))
    return euler_error, improved_error

# Streamlit App Design
st.set_page_config(page_title="Numerical Methods Solver", layout="wide")
st.title("Numerical Methods Solver")
st.sidebar.title("Guide")
st.sidebar.write("### Navigation")
st.sidebar.write("Navigate through the tabs to explore numerical methods.")
st.sidebar.write("### Contributors")
st.sidebar.write("Developed by  Muhammad Basil Irfan, Shahood Rehan and Muhammad Arqam")

# Tabs for different sections
selected_tab = st.tabs(["Euler Methods", "Simpson's Rule", "Compare Results", "Help"])



# Euler and Improved Euler Methods
with selected_tab[0]:
    st.header("Euler and Improved Euler Method Solver")
    
    raw_diff_eq = st.text_input(
        "Enter the differential equation (e.g., \"y' + 2*y = x**3 * np.exp(-2*x)\"):",
        key="diff_eq_input",
    )
    try:
        # Sanitize and format the equation
        sanitized_eq = sanitize_and_format_equation(raw_diff_eq)
        
        # Split the equation into left and right parts
        if "=" in sanitized_eq:
            lhs, rhs = sanitized_eq.split("=")
            diff_eq = f"({rhs.strip()}) - ({lhs.strip()})"  # Represents y' = f(x, y)
        else:
            diff_eq = sanitized_eq
        
        # Input other parameters
        x0 = st.number_input("Initial x (x0):", value=0.0, key="x0_input")
        y0 = st.number_input("Initial y (y0):", value=1.0, key="y0_input")
        h = st.number_input("Step size (h):", value=0.1, key="h_input")
        steps = st.number_input("Number of steps:", value=10, step=1, key="steps_input")
    
        # Display sanitized equation for confirmation
        st.write(f"Sanitized Differential Equation: {diff_eq}")
    except Exception as e:
        st.error(f"Error parsing the equation: {e}")

    
    if st.button("Solve ODE", key="euler_solve"):
        try:
            if not diff_eq or x0 is None or y0 is None or h is None or steps is None:
                st.error("Please ensure all input fields are filled correctly.")
            else:
                diff_eq = format_equation(diff_eq)  # Ensure proper formatting
                x_euler, y_euler = euler_method(diff_eq, x0, y0, h, steps)
                x_improved, y_improved = improved_euler_method(diff_eq, x0, y0, h, steps)
    
                st.write("Euler Method Results:")
                for i, (x, y) in enumerate(zip(x_euler, y_euler)):
                    st.write(f"Step {i}: x = {x:.3f}, y = {y:.3f}")
    
                st.write("Improved Euler Method Results:")
                for i, (x, y) in enumerate(zip(x_improved, y_improved)):
                    st.write(f"Step {i}: x = {x:.3f}, y = {y:.3f}")
        except ValueError as ve:
            st.error(f"Error solving the ODE: {ve}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    
        st.subheader("Results")
        euler_df = pd.DataFrame({"x": x_euler, "Euler y": y_euler})
        improved_df = pd.DataFrame({"x": x_improved, "Improved Euler y": y_improved})
        combined_df = pd.merge(euler_df, improved_df, on="x")
        st.write(combined_df)
    
        fig, ax = plt.subplots()
        ax.plot(x_euler, y_euler, label="Euler Method", marker='o')
        ax.plot(x_improved, y_improved, label="Improved Euler Method", marker='x')
        ax.legend()
        ax.set_title("Comparison of Euler Methods")
        st.pyplot(fig)

        # Error Analysis
        st.subheader("Error Analysis")
        analytical_solution = st.text_input(
            "Enter the analytical solution (e.g., 'np.exp(-x)'):", "np.exp(-x)", key="analytical_solution_input"
        )

        if analytical_solution:
            true_values = [
                eval(analytical_solution, {"x": x, "np": np}) for x in x_euler
            ]
            euler_error = np.abs(np.array(true_values) - np.array(y_euler))
            improved_error = np.abs(np.array(true_values) - np.array(y_improved))

            error_df = pd.DataFrame({
                "x": x_euler,
                "True Value": true_values,
                "Euler Error": euler_error,
                "Improved Euler Error": improved_error,
            })
            st.write(error_df)

            # Plot error comparison
            fig, ax = plt.subplots()
            ax.plot(x_euler, euler_error, label="Euler Error", color='red', linestyle='--')
            ax.plot(x_improved, improved_error, label="Improved Euler Error", color='blue', linestyle='-.')
            ax.set_title("Error Analysis for Euler Methods")
            ax.legend()
            st.pyplot(fig)

# Simpson's Rule
# Streamlit UI for Numerical Integration
with selected_tab[1]:
    st.header("Numerical Integration using Simpson's Rule")
  
    st.header("Numerical Integration using Simpson's Rule (Manual Input)")

    # Input fields for x and f(x)
    st.write("Enter the data points below:")
    x_values = st.text_input("Enter x values (comma-separated, e.g., 0.0, 0.1, 0.2):")
    y_values = st.text_input("Enter f(x) values (comma-separated, e.g., 1.0, 0.9975, 0.99):")

    if x_values and y_values:
        try:
            # Parse the input values
            x_list = list(map(float, x_values.split(",")))
            y_list = list(map(float, y_values.split(",")))

            if len(x_list) != len(y_list):
                st.error("The number of x values and f(x) values must be equal.")
            elif len(x_list) < 3:
                st.error("At least three points are required for numerical integration.")
            else:
                # Combine x and f(x) into points
                points = list(zip(x_list, y_list))

                # Sort points by x-value to ensure consistency
                points = sorted(points, key=lambda x: x[0])

                # Determine which rule to use
                n = len(points) - 1
                if n % 2 == 0:
                    result, method = simpsons_one_third_rule(points)
                elif n % 3 == 0:
                    result, method = simpsons_three_eighths_rule(points)
                else:
                    result, method = None, (
                        "The number of intervals is not suitable for Simpson's 1/3 or 3/8 Rules."
                    )

                # Display results
                if result is not None:
                    st.success(f"Integration Result: {result:.4f}")
                    st.write(f"Method Used: {method}")
                       
                    # Plot x vs f(x)
                    fig, ax = plt.subplots()
                    ax.plot(x_list, y_list, label="f(x) values", marker='o')
                    ax.set_title(f"Numerical Integration using {method}")
                    ax.set_xlabel("x values")
                    ax.set_ylabel("f(x) values")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.error(method)

        except Exception as e:
            st.error(f"Error processing the input data: {e}")
    



# Compare Results
with selected_tab[2]:
    st.header("Compare Results of Euler Method and Simpson's Rule")
    
    # Collect inputs for both methods
    st.subheader("Simpson's Rule Results")
    
    x_values = st.text_input("Enter x values for Simpson's Rule (comma-separated, e.g., 0.0, 0.1, 0.2):")
    y_values = st.text_input("Enter f(x) values for Simpson's Rule (comma-separated, e.g., 1.0, 0.9975, 0.99):")

    simpson_result = None
    if x_values and y_values:
        try:
            # Process Simpson's input data
            x_list = list(map(float, x_values.split(",")))
            y_list = list(map(float, y_values.split(",")))

            if len(x_list) != len(y_list):
                st.error("The number of x values and f(x) values must be equal.")
            elif len(x_list) < 3:
                st.error("At least three points are required for numerical integration.")
            else:
                points = list(zip(x_list, y_list))
                points = sorted(points, key=lambda x: x[0])

                n = len(points) - 1
                if n % 2 == 0:
                    simpson_result, _ = simpsons_one_third_rule(points)
                elif n % 3 == 0:
                    simpson_result, _ = simpsons_three_eighths_rule(points)

        except Exception as e:
            st.error(f"Error processing Simpson's Rule: {e}")

    st.subheader("Euler Method Results")

    # Input for Euler method
    raw_diff_eq = st.text_input("Enter the differential equation for Euler method:", key="diff_eq_input_euler")
    if raw_diff_eq:
        try:
            sanitized_eq = sanitize_and_format_equation(raw_diff_eq)
            if "=" in sanitized_eq:
                lhs, rhs = sanitized_eq.split("=")
                diff_eq = f"({rhs.strip()}) - ({lhs.strip()})"
            else:
                diff_eq = sanitized_eq

            x0 = st.number_input("Initial x (x0):", value=0.0)
            y0 = st.number_input("Initial y (y0):", value=1.0)
            h = st.number_input("Step size (h):", value=0.1)
            steps = st.number_input("Number of steps:", value=10)

            if st.button("Solve ODE for Euler", key="euler_solve_compare"):
                x_euler, y_euler = euler_method(diff_eq, x0, y0, h, steps)

                # Display Euler results
                st.write("Euler Method Results:")
                for i, (x, y) in enumerate(zip(x_euler, y_euler)):
                    st.write(f"Step {i}: x = {x:.3f}, y = {y:.3f}")
                
                # Compare Euler's and Simpson's result
                if simpson_result is not None:
                    st.write(f"Simpson's Rule Result: {simpson_result:.4f}")
                    st.write("Compare the values at each step to see the difference between methods.")

                # Create comparison plot (if meaningful)
                if simpson_result is not None:
                    fig, ax = plt.subplots()
                    ax.plot(x_euler, y_euler, label="Euler Method", marker='o')
                    ax.plot(x_list, y_list, label="Simpson's Rule Approximation", marker='x')
                    ax.legend()
                    ax.set_title("Comparison of Euler Method and Simpson's Rule")
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing the Euler method: {e}")

# Help Section
with selected_tab[3]:
    st.header("Help and Methodology")

    # Euler Method
    st.write("### Euler Method")
    st.write("""
    The Euler Method is a first-order numerical procedure for solving ordinary differential equations (ODEs) with a given initial value.
    It estimates the solution of an ODE by incrementally moving in small steps based on the slope at the current point.
    """)
    st.write("#### **Algorithm**")
    st.write("""
    1. Start with the initial condition \( y_0 = y(x_0) \).
    2. Divide the interval \([x_0, x_n]\) into \(n\) subintervals, each of size \( h \) (step size).
    3. For each step, calculate: 
       \[
       y_{i+1} = y_i + h \cdot f(x_i, y_i)
       \]
    """)
    st.write("#### **Step-by-Step Working**")
    st.write("""
    1. Define the initial values \( x_0, y_0, \) and step size \( h \).
    2. Compute the slope \( f(x_i, y_i) \) at the current point.
    3. Update \( y \) using the formula \( y_{i+1} = y_i + h \cdot f(x_i, y_i) \).
    4. Repeat until the desired interval is covered.
    """)
    st.write("#### **Advantages and Limitations**")
    st.write("""
    **Advantages**: Simple and easy to implement. Works well for small step sizes.\n
    **Limitations**: Low accuracy for larger step sizes, and accumulated error can become significant.
    """)

    # Improved Euler Method
    st.write("### Improved Euler Method (Heun's Method)")
    st.write("""
    Also known as Heun's Method, this is a second-order method that refines the slope estimation, making it more accurate than the Euler Method.
    """)
    st.write("#### **Algorithm**")
    st.write("""
    1. Compute the initial slope: 
       \[
       k_1 = f(x_i, y_i)
       \]
    2. Predict the next value:
       \[
       y_{\text{predicted}} = y_i + h \cdot k_1
       \]
    3. Compute the slope at the predicted point:
       \[
       k_2 = f(x_i + h, y_{\text{predicted}})
       \]
    4. Update the value of \( y \) using the average slope:
       \[
       y_{i+1} = y_i + \frac{h}{2} (k_1 + k_2)
       \]
    """)
    st.write("#### **Advantages and Limitations**")
    st.write("""
    **Advantages**: Higher accuracy compared to the Euler Method while remaining relatively simple.\n
    **Limitations**: Requires additional computation for the second slope.
    """)

    # Simpson's Rule
    st.write("### Simpson's Rule")
    st.write("""
    Simpson's Rule is a numerical method for approximating the definite integral of a function by using polynomials to model the function.
    """)
    st.write("#### **Simpson's 1/3rd Rule**")
    st.write("""
    This rule applies parabolic arcs to approximate the curve between data points. It is more accurate for even intervals.
    **Algorithm**:
    1. Divide the interval into an even number of subintervals.
    2. Apply the formula:
       \[
       \int_a^b f(x) dx \approx \frac{h}{3} \left[ f(x_0) + 4 \sum_{i \text{ odd}} f(x_i) + 2 \sum_{i \text{ even}} f(x_i) + f(x_n) \right]
       \]
       where \( h = \frac{b-a}{n} \).
    """)

    st.write("#### **Simpson's 3/8th Rule**")
    st.write("""
    This rule uses cubic arcs for approximation and is applied when the number of intervals is not divisible by 2.
    **Algorithm**:
    1. Divide the interval into subintervals, ensuring the number of intervals is a multiple of 3.
    2. Apply the formula:
       \[
       \int_a^b f(x) dx \approx \frac{3h}{8} \left[ f(x_0) + 3 \sum_{i \equiv 1,2 \mod 3} f(x_i) + 2 \sum_{i \equiv 0 \mod 3, i \neq 0} f(x_i) + f(x_n) \right]
       \]
    """)

    st.write("#### **Advantages and Limitations**")
    st.write("""
    **Advantages**: High accuracy for smooth functions. Works well for definite integrals.\n
    **Limitations**: Requires evenly spaced intervals and slightly complex computations compared to Euler methods.
    """)

    # Comparison Table
    st.write("### Comparison of Methods")
    st.table({
        "Method": ["Euler", "Improved Euler", "Simpson's 1/3rd", "Simpson's 3/8th"],
        "Accuracy": ["Low", "Moderate", "High (even steps)", "High (odd steps)"],
        "Ease of Use": ["Easy", "Moderate", "Moderate", "Moderate"],
        "Applications": [
            "Quick estimations, learning basics",
            "Higher accuracy with less complexity",
            "Definite integrals of smooth functions",
            "Definite integrals for odd intervals"
        ]
    })

    st.write("#### Recommendations")
    st.write("""
    - **Euler Method**: Use for quick and rough approximations.
    - **Improved Euler Method**: Use when higher accuracy is needed without significant computational complexity.
    - **Simpson's Rule**: Use for solving definite integrals, choosing 1/3rd Rule for even intervals and 3/8th Rule for odd intervals.
    """)

# Footer
st.write("---")
st.write("Developed by  Muhammad Basil Irfan, Shahood Rehan and Muhammad Arqam")
