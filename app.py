
import streamlit as st
import pandas as pd
import numpy as np
from core import ask_question, run_code


# ---------------------------------
# RESULT VALIDATION FUNCTION
# ---------------------------------

def validate_result(result):

    warnings = []

    # Check for empty results
    if result is None:
        warnings.append("AI returned no result.")

    elif isinstance(result, (pd.DataFrame, pd.Series)):

        if result.empty:
            warnings.append("The analysis returned an empty result.")

        # Check for missing values in the result
        if result.isnull().values.any() if isinstance(result, pd.DataFrame) else result.isnull().any():
            warnings.append("The result contains missing values.")

        # Check for infinite numeric values
        numeric_data = (
            result.select_dtypes(include="number")
            if isinstance(result, pd.DataFrame)
            else result
        )

        if isinstance(numeric_data, pd.DataFrame):
            if np.isinf(numeric_data.to_numpy()).any():
                warnings.append("The result contains infinite numeric values.")

        elif pd.api.types.is_numeric_dtype(numeric_data):
            if np.isinf(numeric_data).any():
                warnings.append("The result contains infinite numeric values.")

    elif isinstance(result, (int, float, np.number)):

        if not np.isfinite(result):
            warnings.append("The result is not a finite number.")

    return warnings


# ---------------------------------
# APPLICATION TITLE
# ---------------------------------

st.title(':bar_chart: Chat with your Data :chart_with_upwards_trend:')


# ---------------------------------
# CSV FILE UPLOAD
# ---------------------------------

file = st.file_uploader(
    'Upload a CSV file',
    type="csv"
)


if file:

    try:

        df = pd.read_csv(file)

        if df.empty:
            st.warning("The uploaded CSV file contains no data.")

        else:

            # ---------------------------------
            # DATASET PREVIEW
            # ---------------------------------

            st.subheader("Dataset Preview")
            st.dataframe(df.head())


            # ---------------------------------
            # DATASET INFORMATION
            # ---------------------------------

            st.subheader("Dataset Information")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Rows", df.shape[0])

            with col2:
                st.metric("Total Columns", df.shape[1])


            # ---------------------------------
            # DATA QUALITY REPORT
            # ---------------------------------

            st.subheader("Data Quality Report")

            missing_values = df.isnull().sum()
            duplicate_rows = df.duplicated().sum()

            col3, col4 = st.columns(2)

            with col3:
                st.metric(
                    "Total Missing Values",
                    int(missing_values.sum())
                )

            with col4:
                st.metric(
                    "Duplicate Rows",
                    int(duplicate_rows)
                )

            quality_report = pd.DataFrame({
                "Data Type": df.dtypes.astype(str),
                "Missing Values": missing_values,
                "Missing Percentage": (
                    (missing_values / len(df)) * 100
                ).round(2)
            })

            quality_report.index.name = "Column"

            st.write("Column-wise Quality Report")

            st.dataframe(
                quality_report.reset_index()
            )


            # ---------------------------------
            # AI DATA ANALYSIS
            # ---------------------------------

            st.subheader("Ask Questions About Your Data")

            question = st.text_input(
                "Ask a question about your dataset"
            )

            if st.button("Get Answer"):

                if question.strip():

                    with st.spinner("Analyzing your data..."):

                        try:

                            # Generate Pandas code
                            code = ask_question(df, question)

                            st.subheader("AI Generated Pandas Code")
                            st.code(code, language="python")


                            # Execute generated code
                            result = run_code(df, code)


                            # Check execution errors
                            if isinstance(result, str) and result.startswith("Error:"):

                                st.error(result)

                            else:

                                # ---------------------------------
                                # AI RESULT VALIDATION
                                # ---------------------------------

                                st.subheader("AI Result Validation")

                                validation_warnings = validate_result(result)

                                if validation_warnings:

                                    for warning in validation_warnings:
                                        st.warning(warning)

                                    st.info(
                                        "Please review the result before relying on it."
                                    )

                                else:

                                    st.success(
                                        "Basic result checks passed."
                                    )

                                st.caption(
                                    "These checks identify basic output issues. "
                                    "They do not independently verify the mathematical "
                                    "correctness of the AI-generated answer."
                                )


                                # ---------------------------------
                                # DISPLAY ANALYSIS RESULT
                                # ---------------------------------

                                st.subheader("Analysis Result")

                                if isinstance(result, (pd.DataFrame, pd.Series)):

                                    st.dataframe(result)

                                else:

                                    st.write(result)


                                # ---------------------------------
                                # AUTOMATIC VISUALIZATION
                                # ---------------------------------

                                if isinstance(result, pd.DataFrame):

                                    numeric_columns = result.select_dtypes(
                                        include="number"
                                    ).columns.tolist()

                                    if numeric_columns and len(result.columns) >= 2:

                                        st.subheader("Data Visualization")

                                        chart_column = st.selectbox(
                                            "Select a column to visualize",
                                            numeric_columns
                                        )

                                        category_column = st.selectbox(
                                            "Select a category column",
                                            [
                                                column
                                                for column in result.columns
                                                if column != chart_column
                                            ]
                                        )

                                        chart_data = result.set_index(
                                            category_column
                                        )[[chart_column]]

                                        st.bar_chart(chart_data)

                                    elif numeric_columns:

                                        st.info(
                                            "The result has numeric values but no "
                                            "separate category column for a bar chart."
                                        )

                                    else:

                                        st.info(
                                            "No numeric columns available for visualization."
                                        )


                                elif isinstance(result, pd.Series):

                                    if pd.api.types.is_numeric_dtype(result):

                                        st.subheader("Data Visualization")

                                        st.bar_chart(result)

                                    else:

                                        st.info(
                                            "The result does not contain numeric values "
                                            "for a chart."
                                        )

                        except Exception as error:

                            st.error(
                                f"Unable to process your question: {error}"
                            )

                else:

                    st.warning("Please enter a question.")


    except Exception as error:

        st.error(
            f"Unable to read the CSV file: {error}"
        )
