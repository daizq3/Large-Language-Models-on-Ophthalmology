##question length
import pandas as pd

# 读取Excel文件
final_gpt_df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

# 计算每个问题的长度并添加为新列
final_gpt_df['question_length'] = final_gpt_df['Question'].str.len()

# 将DataFrame保存回Excel文件
final_gpt_df.to_excel('final_gpt35_gpt4_2.xlsx', index=False)

import pandas as pd

#%%
#table n
import pandas as pd
df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

# Perform the required analysis
# 1. Count the total number of questions
total_questions = df['Question'].count()

# 2. Count the number of questions per category
category_counts = df['Category'].value_counts().to_dict()

# 3. Count the number of Lower and Higher classification questions based on gpt-4_classification
classification_counts = df['gpt-4_classification'].value_counts().to_dict()

print("Total Questions:", total_questions)
print("Questions per Category:", category_counts)
print("Classification Counts:", classification_counts)

#%%
#table
import pandas as pd


df = pd.read_excel('final_gpt35_gpt4_2.xlsx')
# Correcting the error by removing the percentage signs and converting the strings to floats
def convert_percentage(perc_str):
    return float(perc_str.strip('%')) / 100

# Apply the conversion function to the 'Percentage' column
df['Percentage'] = df['Percentage'].apply(convert_percentage)

# Calculate the mean percentage and standard deviation for 'All questions'
all_questions_mean = df['Percentage'].mean()
all_questions_std = df['Percentage'].std()

# Calculate the mean percentage and standard deviation for 'Higher' and 'Lower' questions
higher_questions_mean = df[df['gpt-4_classification'] == 'Higher']['Percentage'].mean()
higher_questions_std = df[df['gpt-4_classification'] == 'Higher']['Percentage'].std()

lower_questions_mean = df[df['gpt-4_classification'] == 'Lower']['Percentage'].mean()
lower_questions_std = df[df['gpt-4_classification'] == 'Lower']['Percentage'].std()

# Calculate the mean percentage and standard deviation for each category
categories_mean_std = df.groupby('Category')['Percentage'].agg(['mean', 'std']).to_dict('index')

# Printing the results
print(f"All Questions Mean %: {all_questions_mean:.1%} (SD: {all_questions_std:.1%})")
print(f"Higher Questions Mean %: {higher_questions_mean:.1%} (SD: {higher_questions_std:.1%})")
print(f"Lower Questions Mean %: {lower_questions_mean:.1%} (SD: {lower_questions_std:.1%})")
for category, stats in categories_mean_std.items():
    print(f"{category} Questions Mean %: {stats['mean']:.1%} ({stats['std']:.1%})")

# Printing the results without the percentage symbol and multiplying by 100 for display purposes
print(f"All Questions Mean: {all_questions_mean * 100:.1f} ({all_questions_std * 100:.1f})")
print(f"Higher Questions Mean: {higher_questions_mean * 100:.1f} ({higher_questions_std * 100:.1f})")
print(f"Lower Questions Mean: {lower_questions_mean * 100:.1f} ({lower_questions_std * 100:.1f})")
for category, stats in categories_mean_std.items():
    print(f"{category} Questions Mean: {stats['mean'] * 100:.1f} ({stats['std'] * 100:.1f})")    



#%%
#table LLM1
import pandas as pd

df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

# Calculate the total number of correct answers
total_correct = df[df['gpt-3.5-turbo_answerletter'] == df['Correct Answer Letter']].shape[0]

# Calculate the total number of questions
total_questions = df.shape[0]

# Calculate the percentage of correct answers
percentage_correct = (total_correct / total_questions) 

# Calculate the number of correct answers for Higher and Lower classification
higher_correct = df[(df['gpt-4_classification'] == 'Higher') & 
                    (df['gpt-3.5-turbo_answerletter'] == df['Correct Answer Letter'])].shape[0]
lower_correct = df[(df['gpt-4_classification'] == 'Lower') & 
                   (df['gpt-3.5-turbo_answerletter'] == df['Correct Answer Letter'])].shape[0]

# Calculate the total number of Higher and Lower questions
total_higher_questions = df[df['gpt-4_classification'] == 'Higher'].shape[0]
total_lower_questions = df[df['gpt-4_classification'] == 'Lower'].shape[0]

# Calculate the percentage of correct answers for Higher and Lower classification
percentage_higher_correct = (higher_correct / total_higher_questions) 
percentage_lower_correct = (lower_correct / total_lower_questions) 

# Calculate the number of correct answers and the percentage for each category
categories_correct = df[df['gpt-3.5-turbo_answerletter'] == df['Correct Answer Letter']].groupby('Category').size()
categories_total = df.groupby('Category').size()
categories_percentage = (categories_correct / categories_total) 


# Printing the results with the format "Number of correct questions (Percentage)"
print(f"Total number of correct questions: {total_correct} ({percentage_correct:.1%})")
print(f"Number of correct 'Higher' questions: {higher_correct} ({percentage_higher_correct:.1%})")
print(f"Number of correct 'Lower' questions: {lower_correct} ({percentage_lower_correct:.1%})")

print("Number and percentage of correct questions by Category:")
for category, correct_count in categories_correct.items():
    print(f"{category}: {correct_count} ({categories_percentage[category]:.1%})")

# Printing the results with the format "Number of correct questions (Percentage)"
print(f"Total number of correct questions: {total_correct} ({percentage_correct* 100:.1f})")
print(f"Number of correct 'Higher' questions: {higher_correct} ({percentage_higher_correct* 100:.1f})")
print(f"Number of correct 'Lower' questions: {lower_correct} ({percentage_lower_correct* 100:.1f})")

print("Number and percentage of correct questions by Category:")
for category, correct_count in categories_correct.items():
    print(f"{category}: {correct_count} ({categories_percentage[category]* 100:.1f})")    


#%%
#table LLM2
import pandas as pd

df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

# Calculate the total number of correct answers
total_correct = df[df['gpt-4_answerletter'] == df['Correct Answer Letter']].shape[0]

# Calculate the total number of questions
total_questions = df.shape[0]

# Calculate the percentage of correct answers
percentage_correct = (total_correct / total_questions) 

# Calculate the number of correct answers for Higher and Lower classification
higher_correct = df[(df['gpt-4_classification'] == 'Higher') & 
                    (df['gpt-4_answerletter'] == df['Correct Answer Letter'])].shape[0]
lower_correct = df[(df['gpt-4_classification'] == 'Lower') & 
                   (df['gpt-4_answerletter'] == df['Correct Answer Letter'])].shape[0]

# Calculate the total number of Higher and Lower questions
total_higher_questions = df[df['gpt-4_classification'] == 'Higher'].shape[0]
total_lower_questions = df[df['gpt-4_classification'] == 'Lower'].shape[0]

# Calculate the percentage of correct answers for Higher and Lower classification
percentage_higher_correct = (higher_correct / total_higher_questions) 
percentage_lower_correct = (lower_correct / total_lower_questions) 

# Calculate the number of correct answers and the percentage for each category
categories_correct = df[df['gpt-4_answerletter'] == df['Correct Answer Letter']].groupby('Category').size()
categories_total = df.groupby('Category').size()
categories_percentage = (categories_correct / categories_total) 


# Printing the results with the format "Number of correct questions (Percentage)"
print(f"Total number of correct questions: {total_correct} ({percentage_correct:.1%})")
print(f"Number of correct 'Higher' questions: {higher_correct} ({percentage_higher_correct:.1%})")
print(f"Number of correct 'Lower' questions: {lower_correct} ({percentage_lower_correct:.1%})")

print("Number and percentage of correct questions by Category:")
for category, correct_count in categories_correct.items():
    print(f"{category}: {correct_count} ({categories_percentage[category]:.1%})")

# Printing the results with the format "Number of correct questions (Percentage)"
print(f"Total number of correct questions: {total_correct} ({percentage_correct* 100:.1f})")
print(f"Number of correct 'Higher' questions: {higher_correct} ({percentage_higher_correct* 100:.1f})")
print(f"Number of correct 'Lower' questions: {lower_correct} ({percentage_lower_correct* 100:.1f})")

print("Number and percentage of correct questions by Category:")
for category, correct_count in categories_correct.items():
    print(f"{category}: {correct_count} ({categories_percentage[category]* 100:.1f})")    


#%%

import pandas as pd

df = pd.read_excel('final_gpt35_gpt4_2.xlsx')
       
# 检查 gpt-3.5 的答案是否正确
df['correct_gpt3.5'] = df['gpt-3.5-turbo_answerletter'] == df['Correct Answer Letter']

# 检查 gpt-4 的答案是否正确
df['correct_gpt4'] = df['gpt-4_answerletter'] == df['Correct Answer Letter']

# 保存修改后的文件
df.to_excel('final_gpt35_gpt4_2.xlsx', index=False)        


#%%
import pandas as pd
from scipy.stats import chi2_contingency
from itertools import combinations


# Define the data as seen in the provided image
data = {
    'Total': {'Human': (467, round(467 * 0.768)), 'LLM 1': (467, 271), 'LLM 2': (467, 371)},
    'Higher': {'Human': (259, round(259 * 0.766)), 'LLM 1': (259, 128), 'LLM 2': (259, 201)},
    'Lower': {'Human': (208, round(208 * 0.772)), 'LLM 1': (208, 143), 'LLM 2': (208, 170)},
}

# Categories data
categories_data = {
    "Basic science": {"Human": (53, round(53 * 0.738)), "LLM 1": (53, 28), "LLM 2": (53, 43)},
    "Cataract and Anterior Segment": {"Human": (59, round(59 * 0.760)), "LLM 1": (59, 30), "LLM 2": (59, 39)},
    "Cornea and External Disease": {"Human": (51, round(51 * 0.794)), "LLM 1": (51, 30), "LLM 2": (51, 44)},
    "Glaucoma": {"Human": (64, round(64 * 0.827)), "LLM 1": (64, 40), "LLM 2": (64, 55)},
    "Neuro-Ophthalmology and Orbit": {"Human": (60, round(60 * 0.737)), "LLM 1": (60, 37), "LLM 2": (60, 47)},
    "Oculoplastics": {"Human": (64, round(64 * 0.770)), "LLM 1": (64, 38), "LLM 2": (64, 45)},
    "Ophthalmic Pathology and Oncology": {"Human": (5, round(5 * 0.646)), "LLM 1": (5, 2), "LLM 2": (5, 3)},
    "Pediatric Ophthalmology and Strabismus": {"Human": (33, round(33 * 0.804)), "LLM 1": (33, 19), "LLM 2": (33, 30)},
    "Retina and Vitreous": {"Human": (28, round(28 * 0.749)), "LLM 1": (28, 15), "LLM 2": (28, 23)},
    "Uveitis": {"Human": (50, round(50 * 0.745)), "LLM 1": (50, 32), "LLM 2": (50, 42)}
}


# This function will perform chi-square tests for all combinations within each group or category
# Adjusted function to perform chi-square tests, with an option to apply Bonferroni correction
def perform_chi_square_tests(group_data, apply_correction=True, correction_factor=3):
    p_values = {}
    # Perform the chi-square test for each pair within the group
    for (group_1, data_1), (group_2, data_2) in combinations(group_data.items(), 2):
        # Create the contingency table
        contingency_table = [list(data_1), list(data_2)]
        # Perform the chi-square test
        chi2, p, _, _ = chi2_contingency(contingency_table)
        # Adjust the p-value for Bonferroni correction if required
        p_adjusted = min(1, p * correction_factor) if apply_correction else p
        # Store the p-value
        p_values[f"{group_1} vs {group_2}"] = p_adjusted
    return p_values



# Perform the chi-square tests for Total, Higher, and Lower without Bonferroni correction
p_values_total_higher_lower = {group: perform_chi_square_tests(group_data, apply_correction=False)
                               for group, group_data in data.items()}

# Assuming the rest of the categories_data is defined similar to the provided structure,
# Perform the chi-square tests for categories with Bonferroni correction
p_values_categories = {category: perform_chi_square_tests(group_data, apply_correction=True)
                       for category, group_data in categories_data.items()}

# Combine all p-values into one dictionary
all_p_values_combined = {**p_values_total_higher_lower, **p_values_categories}



# Format the p-values as specified
for group, p_values in all_p_values_combined.items():
    all_p_values_combined[group] = {}
    for comparison, p_value in p_values.items():
        if p_value < 0.001:
            formatted_p = "<.001"
        elif p_value > 0.99:
            formatted_p = ">.99"
        elif 0.001 <= p_value < 0.01:
            # Report to the nearest thousandth for p-values between 0.001 and 0.01
            formatted_p = f"{p_value:.3f}"
        else:
            # For p-values >= 0.01, report to the nearest hundredth
            formatted_p = f"{p_value:.2f}"
        all_p_values_combined[group][comparison] = formatted_p

# Now let's print the formatted p-values
for group, comparisons in all_p_values_combined.items():
    print(f"{group}:")
    for comparison, p_value in comparisons.items():
        print(f"  {comparison}: {p_value}")
    print()  # Print a newline for better readability between groups

#%%
#计算矫正后的正确率，并比较
import pandas as pd


df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

# 将correct_gpt3.5列中的TRUE/FALSE替换为1/0
df['correct_gpt3.5'] = df['correct_gpt3.5'].map({True: 1, False: 0})

# 将correct_gpt4列中的TRUE/FALSE替换为1/0
df['correct_gpt4'] = df['correct_gpt4'].map({True: 1, False: 0})

# 保存更改后的Excel文件
df.to_excel('final_gpt35_gpt4_2.xlsx', index=False)


#%%
#计算矫正后的正确率，并比较
import pandas as pd
from scipy import stats

def calculate_guess_corrected_scores(df, column_name, num_choices):
    # 计算正确和错误的回答数量
    num_correct = df[column_name].sum()
    num_incorrect = len(df) - num_correct

    # 应用猜测校正公式
    guess_corrected_score = num_correct - (num_incorrect / (num_choices - 1))
    return guess_corrected_score

# 假设您已经加载了包含相应列的DataFrame
df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

# 对GPT-3.5和GPT-4应用该函数
guess_corrected_score_gpt35 = calculate_guess_corrected_scores(df, 'correct_gpt3.5', 4)
guess_corrected_score_gpt4 = calculate_guess_corrected_scores(df, 'correct_gpt4', 4)

guess_corrected_score_gpt35p=guess_corrected_score_gpt35 / len(df)
guess_corrected_score_gpt4p=guess_corrected_score_gpt4 / len(df)



# 计算GPT-3.5和GPT-4的猜测校正得分
df['guess_corrected_score_gpt35'] = df['correct_gpt3.5'].apply(lambda x: calculate_guess_corrected_scores(df, 'correct_gpt3.5', 4))
df['guess_corrected_score_gpt4'] = df['correct_gpt4'].apply(lambda x: calculate_guess_corrected_scores(df, 'correct_gpt4', 4))

# 执行t检验
t_stat, p_value = stats.ttest_ind(df['guess_corrected_score_gpt35'], df['guess_corrected_score_gpt4'])

print("guess_corrected_score_gpt35:", guess_corrected_score_gpt35)
print("guess_corrected_score_gpt4:", guess_corrected_score_gpt4)
print("guess_corrected_score_gpt35p:", guess_corrected_score_gpt35p)
print("guess_corrected_score_gpt4p:", guess_corrected_score_gpt4p)
print("t_stat:", t_stat)
print("p_value:", format(p_value, '.3f'))  # 将p值保留到小数点后三位


#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from the table provided by the user
data = {
    'Metric': ['All questions', 'Higher order thinking', 'Lower order thinking', 
               'Basic science', 'Cataract and Anterior Segment', 
               'Cornea and External Disease', 'Glaucoma', 'Neuro-Ophthalmology', 
               'Oculoplastics', 'Ophthalmic Pathology and Oncology', 
               'Pediatric Ophthalmology and Strabismus', 'Retina and Vitreous', 'Uveitis'],
    'Human': [76.8, 76.6, 77.2, 73.8, 76.0, 79.4, 82.7, 73.7, 77.0, 64.6, 80.4, 74.9, 74.5],
    'LLM1': [58.0, 49.4, 68.8, 52.8, 50.8, 58.8, 62.5, 61.7, 59.4, 40.0, 57.6, 53.6, 64.0],
    'LLM2': [79.4, 77.6, 81.7, 81.1, 66.1, 86.3, 85.9, 78.3, 70.3, 60.0, 90.9, 82.1, 84.0]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Number of variables
categories = list(df['Metric'])
N = len(categories)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1] # to close the plot

# Initialise the radar plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories, color='grey', size=8)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([20,40,60,80], ["20","40","60","80"], color="grey", size=7)
plt.ylim(0,100)

# Plot each individual = each line of the data
# I add a 'group' column, then call plot for each value of that column:
for column in df.drop('Metric', axis=1):
    values=df[column].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=column)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Show the plot
plt.show()
#%%
#雷达图
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import pi

# Data from the table provided by the user
data = {
    'Metric': ['All questions', 'Higher order thinking', 'Lower order thinking', 
               'Basic science', 'Cataract and Anterior Segment', 
               'Cornea and External Disease', 'Glaucoma', 'Neuro-Ophthalmology', 
               'Oculoplastics', 'Ophthalmic Pathology and Oncology', 
               'Pediatric Ophthalmology and Strabismus', 'Retina and Vitreous', 'Uveitis'],
    'Human': [76.8, 76.6, 77.2, 73.8, 76.0, 79.4, 82.7, 73.7, 77.0, 64.6, 80.4, 74.9, 74.5],
    'LLM1': [58.0, 49.4, 68.8, 52.8, 50.8, 58.8, 62.5, 61.7, 59.4, 40.0, 57.6, 53.6, 64.0],
    'LLM2': [79.4, 77.6, 81.7, 81.1, 66.1, 86.3, 85.9, 78.3, 70.3, 60.0, 90.9, 82.1, 84.0]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Number of variables
categories = list(df['Metric'])
N = len(categories)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1] # to close the plot

# Initialise the spider plot
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories, color='black', size=16)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="black", size=14)
plt.ylim(0,100)

# Plot each individual = each line of the data
# Define colors for better contrast
#colors = ['blue', 'red', 'green'] # Blue for Human, Red for LLM1, Green for LLM2
colors = ['#ff7700', '#00aaff', '#33cc33'] 
# Plot each individual = each line of the data with new colors
for idx, column in enumerate(df.drop('Metric', axis=1)):
    values = df[column].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=column, color=colors[idx])

# Add legend
plt.legend(loc='lower right', bbox_to_anchor=(0.015, 0.015),fontsize=14)

plt.savefig('C:/Users/子琦/Desktop/LLM/figure_table/rader_graph.pdf', format='pdf')
plt.savefig('C:/Users/子琦/Desktop/LLM/figure_table/rader_graph.svg', format='svg')

# Show the plot
plt.show()


#%%
#etable2counts
import pandas as pd

df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

def convert_percentage(perc):
    if isinstance(perc, str):
        return float(perc.strip('%')) / 100
    return perc

df['Percentage'] = df['Percentage'].apply(convert_percentage)


# Calculate quartiles inversely (lower percentage means more difficult)
df['Quartile'] = pd.qcut(df['Percentage'], 4, labels=[
    'Difficult Questions (4th quartile)',  # 1st quartile (lowest percentages)
    'Advanced Questions (3rd quartile)',   # 2nd quartile
    'Intermediate Questions (2nd quartile)',# 3rd quartile
    'Easy Questions (1st quartile)'        # 4th quartile (highest percentages)
])

# Determine the Higher or Lower classification within each quartile
df['Classification'] = df['gpt-4_classification'].apply(lambda x: 'Higher' if x == 'Higher' else 'Lower')

# Count the number of questions in each category
question_counts = df.groupby(['Quartile', 'Classification']).size().reset_index(name='Question Count')

# Print the results
print(question_counts)

#%%
#etable2
import pandas as pd

df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

def convert_percentage(perc):
    if isinstance(perc, str):
        return float(perc.strip('%')) / 100
    return perc

# 应用转换百分比函数
df['Percentage'] = df['Percentage'].apply(convert_percentage)

# 定义四分位数标签
quartile_labels = [
    'Easy Questions (1st quartile)', 
    'Intermediate Questions (2nd quartile)',
    'Advanced Questions (3rd quartile)', 
    'Difficult Questions (4th quartile)'
]

# 基于反转的 'Percentage' 值计算四分位数
df['Quartile'] = pd.qcut(df['Percentage'], 4, labels=quartile_labels[::-1])

# 计算每个四分位数和分类中 'Percentage' 的平均值和标准差
quartile_classification_means = df.groupby(['Quartile', 'gpt-4_classification'])['Percentage'].mean().unstack()
quartile_classification_std = df.groupby(['Quartile', 'gpt-4_classification'])['Percentage'].std().unstack()

# 创建新的 DataFrame 以保存格式化的平均值（标准差）值
formatted_means_sd = quartile_classification_means.copy()

# 遍历每个四分位数和分类以格式化输出
for quartile in quartile_labels[::-1]:  # 确保循环的顺序与 DataFrame 索引的顺序相同
    for classification in ['Higher', 'Lower']:
        # 获取当前四分位数和分类的平均值和标准差
        mean = quartile_classification_means.loc[quartile, classification] * 100  # 转换为百分比
        std = quartile_classification_std.loc[quartile, classification] * 100     # 转换为百分比
        # 将平均值和标准差格式化为 "mean (sd)"
        formatted_means_sd.loc[quartile, classification] = f"{mean:.1f} ({std:.1f})"

# 打印格式化的 DataFrame
print(formatted_means_sd)


#%%
#    etable2LLM1
import pandas as pd

df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

def convert_percentage(perc):
    if isinstance(perc, str):
        return float(perc.strip('%')) / 100
    return perc

df['Percentage'] = df['Percentage'].apply(convert_percentage)




# Calculate quartiles based on 'Percentage' values
df['Quartile'] = pd.qcut(df['Percentage'], 4, labels=[
    'Difficult Questions (4th quartile)',  # Lowest percentages, more difficult
    'Advanced Questions (3rd quartile)',   # Second lowest percentages
    'Intermediate Questions (2nd quartile)', # Second highest percentages
    'Easy Questions (1st quartile)'         # Highest percentages, easier
])

# Initialize a dictionary to store correct counts and percentages
correct_counts = {}
correct_percentages = {}

# Calculate the number of correct answers and the percentage for each quartile and classification
for quartile in df['Quartile'].unique():
    for classification in ['Higher', 'Lower']:
        correct = df[(df['Quartile'] == quartile) & 
                     (df['gpt-4_classification'] == classification) & 
                     (df['gpt-3.5-turbo_answerletter'] == df['Correct Answer Letter'])].shape[0]
        total = df[(df['Quartile'] == quartile) & 
                   (df['gpt-4_classification'] == classification)].shape[0]
        
        correct_counts[(quartile, classification)] = correct
        correct_percentages[(quartile, classification)] = correct / total if total > 0 else 0

for (quartile, classification), correct in correct_counts.items():
    percentage = correct_percentages[(quartile, classification)]
    print(f"{quartile} - {classification}: {correct}({percentage:.1%})")

# Output the results
for (quartile, classification), correct in correct_counts.items():
    percentage = correct_percentages[(quartile, classification)]
    print(f"{quartile} - {classification}: {correct}({percentage* 100:.1f})")


#    etable2LLM2
import pandas as pd

df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

def convert_percentage(perc):
    if isinstance(perc, str):
        return float(perc.strip('%')) / 100
    return perc

df['Percentage'] = df['Percentage'].apply(convert_percentage)




# Calculate quartiles based on 'Percentage' values
df['Quartile'] = pd.qcut(df['Percentage'], 4, labels=[
    'Difficult Questions (4th quartile)',  # Lowest percentages, more difficult
    'Advanced Questions (3rd quartile)',   # Second lowest percentages
    'Intermediate Questions (2nd quartile)', # Second highest percentages
    'Easy Questions (1st quartile)'         # Highest percentages, easier
])

# Initialize a dictionary to store correct counts and percentages
correct_counts = {}
correct_percentages = {}

# Calculate the number of correct answers and the percentage for each quartile and classification
for quartile in df['Quartile'].unique():
    for classification in ['Higher', 'Lower']:
        correct = df[(df['Quartile'] == quartile) & 
                     (df['gpt-4_classification'] == classification) & 
                     (df['gpt-4_answerletter'] == df['Correct Answer Letter'])].shape[0]
        total = df[(df['Quartile'] == quartile) & 
                   (df['gpt-4_classification'] == classification)].shape[0]
        
        correct_counts[(quartile, classification)] = correct
        correct_percentages[(quartile, classification)] = correct / total if total > 0 else 0

for (quartile, classification), correct in correct_counts.items():
    percentage = correct_percentages[(quartile, classification)]
    print(f"{quartile} - {classification}: {correct}({percentage:.1%})")

# Output the results
for (quartile, classification), correct in correct_counts.items():
    percentage = correct_percentages[(quartile, classification)]
    print(f"{quartile} - {classification}: {correct}({percentage* 100:.1f})")



#%%
#VS VS VS
import pandas as pd

from scipy.stats import chi2_contingency
from itertools import combinations

# Redefine the quartile data using the actual number of correct answers for LLM 1 and LLM 2 as provided in the table
quartile_data_actual_numbers = {
    "Easy Questions Higher": {"Human": (55, round(55 * 0.939)), "LLM 1": (55, 43), "LLM 2": (55, 53)},
    "Easy Questions Lower": {"Human": (53, round(53 * 0.933)), "LLM 1": (53, 48), "LLM 2": (53, 52)},
    "Intermediate Questions Higher": {"Human": (66, round(66 * 0.845)), "LLM 1": (66, 35), "LLM 2": (66, 55)},
    "Intermediate Questions Lower": {"Human": (45, round(45 * 0.847)), "LLM 1": (45, 34), "LLM 2": (45, 39)},
    "Advanced Questions Higher": {"Human": (72, round(2 * 0.756)), "LLM 1": (72, 28), "LLM 2": (72, 57)},
    "Advanced Questions Lower": {"Human": (56, round(56 * 0.757)), "LLM 1": (56, 30), "LLM 2": (56, 46)},
    "Difficult Questions Higher": {"Human": (66, round(66 * 0.553)), "LLM 1": (66, 22), "LLM 2": (66, 36)},
    "Difficult Questions Lower": {"Human": (54, round(54 * 0.567)), "LLM 1": (54, 31), "LLM 2": (54, 33)}
}
def perform_chi_square_tests(group_data):
    p_values = {}
    # Perform the chi-square test for each pair within the group
    for (group_1, data_1), (group_2, data_2) in combinations(group_data.items(), 2):
        # Create the contingency table
        contingency_table = [list(data_1), list(data_2)]
        # Perform the chi-square test
        chi2, p, _, _ = chi2_contingency(contingency_table)
        # Adjust the p-value for Bonferroni correction
        p_adjusted = min(1, p * 3)  # multiplying by 3 for Bonferroni correction
        # Store the p-value
        p_values[f"{group_1} vs {group_2}"] = p_adjusted
    return p_values

# Perform the chi-square tests for all quartile-based questions and apply Bonferroni correction
quartile_p_values_actual_numbers = {
    group: perform_chi_square_tests(group_data) for group, group_data in quartile_data_actual_numbers.items()
}

# Format the p-values as specified
formatted_quartile_p_values_corrected = {}
for group, p_values in quartile_p_values_actual_numbers.items():
    formatted_quartile_p_values_corrected[group] = {}
    for comparison, p_value in p_values.items():
        if p_value < 0.001:
            formatted_p = "<.001"
        elif p_value > 0.99:
            formatted_p = ">.99"
        else:
            formatted_p = f"{p_value:.2f}"
        formatted_quartile_p_values_corrected[group][comparison] = formatted_p

print(formatted_quartile_p_values_corrected)


#%%
#higher-order questions 
import numpy as np
import pandas as pd
from scipy import stats

# Data from the image provided
human_correct_mean_pct = np.array([93.9, 84.5, 75.6, 55.3])
llm1_correct_pct = np.array([78.2, 53.0, 38.9, 33.3])
llm2_correct_pct = np.array([96.4, 83.3, 79.2, 54.5])

# We can use Pearson's correlation coefficient to determine the correlation
# between the models' performances and the human users' average performance.
r_llm1, p_value_llm1 = stats.pearsonr(human_correct_mean_pct, llm1_correct_pct)
r_llm2, p_value_llm2 = stats.pearsonr(human_correct_mean_pct, llm2_correct_pct)

print(r_llm1, p_value_llm1, r_llm2, p_value_llm2)

#%%
#lower-order questions 
import numpy as np
import pandas as pd
from scipy import stats

# Data from the image provided
human_correct_mean_pct = np.array([93.3, 84.7, 75.7, 56.7])
llm1_correct_pct = np.array([90.6, 75.6, 53.6, 57.4])
llm2_correct_pct = np.array([98.1, 86.7, 82.1, 61.1])

# We can use Pearson's correlation coefficient to determine the correlation
# between the models' performances and the human users' average performance.
r_llm1, p_value_llm1 = stats.pearsonr(human_correct_mean_pct, llm1_correct_pct)
r_llm2, p_value_llm2 = stats.pearsonr(human_correct_mean_pct, llm2_correct_pct)

print(r_llm1, p_value_llm1, r_llm2, p_value_llm2)



#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np



# Revised performance data for plotting with LLM 2 included
performance_data = {
    'LLM 1 Higher': [78.2, 53.0, 38.9, 33.3],  
    'LLM 1 Lower': [90.6, 75.6, 53.6, 57.4],  
    'Human Higher': [93.9, 84.5, 75.6, 55.3],  
    'Human Lower': [93.3, 84.7, 75.7, 56.7],  
    'LLM 2 Higher': [96.4, 83.3, 79.2, 54.5],  
    'LLM 2 Lower': [98.1, 86.7, 82.1, 61.1],  
}

# Custom difficulty level labels with line breaks for the x-axis
difficulty_levels = ['Easy', 'Intermediate', 
                            'Advanced', 'Difficult']

# Setting custom difficulty level labels
plt.xticks(range(len(difficulty_levels)), difficulty_levels, ha='center')

# Colors and markers pre-defined
# Updated colors for a fresher look
#colors = ['#34a853', '#4285f4', '#ea4335']  # Google的品牌色调，清新且现代
colors = ['#00aaff', '#ff7700', '#33cc33']  # 明亮的天蓝色，活泼的橙色，现代感的鲜绿色
#colors = ['#3498db', '#2ecc71', '#f39c12']  # 淡蓝色，柔和的绿色，以及暖橙色
#colors = ['#e66101', '#fdb863', '#b2abd2']  # 亮橙色，浅橙色和柔和的蓝色
# Markers pre-defined
#colors = ['#1f77b4', '#ff7f0e', '#8c564b']  # 橄榄绿，蓝色，橙色,色盲友好
#colors = ['#34495E','#7F8C8D', '#D35400']
#colors = ['#34495E', '#E67E22', '#BDC3C7']
#colors = ['#2CA9E1', '#F7DC6F', '#FA8072']


markers = {'Higher': 'o', 'Lower': 's'}  # Marker styles

# Plotting the performance data with the y-axis starting at 0
plt.figure(figsize=(9.5, 6))

# Plot each line with the specified color and marker
for idx, (group, performance) in enumerate(performance_data.items()):
    color = colors[idx // 2]
    marker = markers['Higher' if 'Higher' in group else 'Lower']
    plt.plot(difficulty_levels, performance, marker=marker, color=color, label=group)

# Adding title and labels with specific font settings
plt.title('Performance of Large Language Models (LLMs) by Difficulty Level', fontdict={'fontsize': 16, 'fontweight': 'bold', 'family': 'Arial'})
plt.xlabel('Difficulty level', fontdict={'fontsize': 16, 'family': 'Arial'})
plt.ylabel('Percentage Correct %', fontdict={'fontsize': 16,'family': 'Arial'})

# Setting the legend with a specific font size
plt.legend(fontsize=14,loc='best')

# Setting the y-axis to start at 0
plt.ylim(0, 100)

# Display the plot with grid
plt.grid(True, axis='y')

# 获取当前的轴
ax = plt.gca()

# 隐藏上边和右边的轴脊
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the plot as a vector image (.svg)
plt.savefig('C:/Users/子琦/Desktop/LLM/figure_table/performance_graph.pdf', format='pdf')
plt.savefig('C:/Users/子琦/Desktop/LLM/figure_table/performance_graph.svg', format='svg')
# Show the plot
plt.show()

#%%
#figure3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取Excel文件
df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

def convert_percentage(perc):
    if isinstance(perc, str):
        return float(perc.strip('%')) 
    return perc

df['Percentage'] = df['Percentage'].apply(convert_percentage)


# Define colors and markers for LLM1 and LLM2
llm_colors = ['#00aaff', '#33cc33']  # Hex colors for LLM1 and LLM2
llm_markers = ['s', 'o']


# Start creating the plot
plt.figure(figsize=(10, 7))

# Create a boxplot with a uniform light blue color for all boxes
sns.boxplot(x='Percentage', y='Category', data=df, whis=1.5, color="lightblue", width=0.4)

# 使用正确的标记形状来绘制 LLM1 和 LLM2 的得分点
for i, llm in enumerate(['correct_gpt3.5', 'correct_gpt4']):
    # Calculate the percentage of correct answers for each LLM within each category
    llm_scores = df.groupby('Category')[llm].mean() * 100
    # Convert series to DataFrame for merging
    llm_scores = llm_scores.reset_index()
    llm_scores.rename(columns={llm: 'Percentage'}, inplace=True)
    # Add scatter plot for LLM scores with specified marker shapes
    sns.scatterplot(
        x='Percentage',
        y='Category',
        data=llm_scores,
        marker=llm_markers[i],
        color=llm_colors[i],
        s=100,
        label=f'LLM {i+1}',
        zorder=5
    )
# Set plot title and labels
plt.title('Accuracy Rate Across Different Topics', fontdict={'fontsize': 16, 'fontweight': 'bold', 'family': 'Arial'})
plt.xlabel('Accuracy Rate %', fontdict={'fontsize': 16,'family': 'Arial'})
plt.ylabel('', fontdict={'fontsize': 16,'family': 'Arial'})

# Adjust legend with specific colors
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels,fontsize='14', title='', handletextpad=1, columnspacing=1, loc='upper left')

# Setting the y-axis to start at 0
plt.xlim(0, 100.1)

# Adjust the spines to show the left spine
ax = plt.gca()
ax.spines['left'].set_visible(True)
ax.spines['left'].set_linewidth(0.5)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['bottom'].set_color('black')
# 隐藏上边和右边的轴脊
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xticks(range(0, 101, 10))



plt.grid(True, 'major', 'x', color='#ddd', linestyle='-', linewidth=0.5, alpha=0.7)
# 确保网格线绘制在图的最前面
plt.gca().set_axisbelow(False)

plt.subplots_adjust(left=0.3)  # 增加左边距的百分比，根据需要调整这个值

# Save the plot as a vector image (.svg)
plt.savefig('C:/Users/子琦/Desktop/LLM/figure_table/Questions_per_Topic_graph.pdf', format='pdf')
plt.savefig('C:/Users/子琦/Desktop/LLM/figure_table/Questions_per_Topic_graph.svg', format='svg')
# Show the final plot

plt.show()


#%%
#Shapiro-Wilk正态性检验LLM1
import scipy.stats as stats
import pandas as pd
df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

# 将信心数据分为正确答案和错误答案两组
confidence_correct = df[df['correct_gpt3.5'] == 1]['gpt-3.5-turbo_confidence_answer_likert']
confidence_incorrect = df[df['correct_gpt3.5'] == 0]['gpt-3.5-turbo_confidence_answer_likert']

# 对正确答案的回答信心进行Shapiro-Wilk检验
w_statistic, p_value = stats.shapiro(confidence_correct)
print("Correct answer Confidence - W statistic:", w_statistic, "P:", p_value)

# 对错误答案的回答信心进行Shapiro-Wilk检验
w_statistic, p_value = stats.shapiro(confidence_incorrect)
print("False answer confidence -W statistic:", w_statistic, "P:", p_value)

#%%
##Shapiro-Wilk正态性检验LLM2
import scipy.stats as stats
import pandas as pd
import pandas as pd
df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

# 将信心数据分为正确答案和错误答案两组
confidence_correct = df[df['correct_gpt4'] == 1]['gpt-4_confidence_answer_likert']
confidence_incorrect = df[df['correct_gpt4'] == 0]['gpt-4_confidence_answer_likert']

# 对正确答案的回答信心进行Shapiro-Wilk检验
w_statistic, p_value = stats.shapiro(confidence_correct)
print("Correct answer Confidence - W statistic:", w_statistic, "P:", p_value)

# 对错误答案的回答信心进行Shapiro-Wilk检验
w_statistic, p_value = stats.shapiro(confidence_incorrect)
print("False answer confidence -W statistic:", w_statistic, "P:", p_value)





#%%
#efigure5 confidence
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# 读取数据
df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

# 重新设置调色板颜色
palette = {0:'#00aaff', 1: '#33cc33' }

# Filter out non-numeric confidence levels and ensure we only include confidence levels 1 through 5
df['gpt-3.5-turbo_confidence_answer_likert'] = pd.to_numeric(df['gpt-3.5-turbo_confidence_answer_likert'], errors='coerce')
df['gpt-4_confidence_answer_likert'] = pd.to_numeric(df['gpt-4_confidence_answer_likert'], errors='coerce')

# Drop rows where confidence level is NaN after coercion
df = df.dropna(subset=['gpt-3.5-turbo_confidence_answer_likert', 'gpt-4_confidence_answer_likert'])

# Ensure that confidence levels are integers
df['gpt-3.5-turbo_confidence_answer_likert'] = df['gpt-3.5-turbo_confidence_answer_likert'].astype(int)
df['gpt-4_confidence_answer_likert'] = df['gpt-4_confidence_answer_likert'].astype(int)

# Set the style of the seaborn plots
sns.set_style("whitegrid", {'axes.grid' : False})

# Set the font properties which will be used for the title and the rest of the plot text
title_font = {'family': 'Arial', 'size': 16, 'weight': 'bold'}
axis_font = {'family': 'Arial', 'size': 16}
legend_font = {'size': 16}

# Apply the font properties to matplotlib
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelweight'] = 'bold'  # Set the weight of the x and y labels


# Re-drawing the plots with corrected x-axis values
plt.figure(figsize=(12, 8))

# GPT-3.5 Confidence Plot
ax1 = plt.subplot(1, 2, 1)
sns.countplot(
    x='gpt-3.5-turbo_confidence_answer_likert', 
    hue='correct_gpt3.5', 
    data=df, 
    palette=palette,
    order=[1, 2, 3, 4, 5]
)
ax1.set_title('LLM1', fontdict=title_font)
ax1.set_xlabel('Confidence', fontdict=axis_font)
ax1.set_ylabel('N Questions', fontdict=axis_font)
ax1.legend(title='Answer Correctness', title_fontsize='16', labels=['Incorrect', 'Correct'], fontsize='16')
ax1.grid(True, linestyle='-', linewidth='0.5', color='#ddd', zorder=1, alpha=0.7)  # Lighter grid lines



# GPT-4 Confidence Plot
ax2 = plt.subplot(1, 2, 2)
sns.countplot(
    x='gpt-4_confidence_answer_likert', 
    hue='correct_gpt4', 
    data=df, 
    palette=palette,
    order=[1, 2, 3, 4, 5]
)
ax2.set_title('LLM2', fontdict=title_font)
ax2.set_xlabel('Confidence', fontdict=axis_font)
ax2.set_ylabel('N Questions', fontdict=axis_font)
ax2.legend(title='Answer Correctness', title_fontsize='16', labels=['Incorrect', 'Correct'], fontsize='16')
ax2.grid(True, linestyle='-', linewidth='0.5', color='#ddd', alpha=0.7, zorder=1)


# Save the current figure to a PDF file
plt.tight_layout()
plt.savefig('C:/Users/子琦/Desktop/LLM/figure_table/Confidence_graph.pdf', format='pdf')
plt.savefig('C:/Users/子琦/Desktop/LLM/figure_table/Confidence_graph.svg', format='svg')
plt.show()
#%%
#GPT3.5 U检验
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu

df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

# Calculate mean and standard deviation (SD) of confidence for correct answers
correct_confidence = df[df['correct_gpt3.5'] == 1]['gpt-3.5-turbo_confidence_answer_likert']
mean_correct_confidence = correct_confidence.mean()
sd_correct_confidence = correct_confidence.std()

# Calculate mean and SD of confidence for incorrect answers
incorrect_confidence = df[df['correct_gpt3.5'] == 0]['gpt-3.5-turbo_confidence_answer_likert']
mean_incorrect_confidence = incorrect_confidence.mean()
sd_incorrect_confidence = incorrect_confidence.std()

print(f"Correct answers - Mean: {mean_correct_confidence}, SD: {sd_correct_confidence}")
print(f"Incorrect answers - Mean: {mean_incorrect_confidence}, SD: {sd_incorrect_confidence}")

# Remove NaN values from incorrect confidence scores
incorrect_confidence_clean = incorrect_confidence.dropna()


# Perform the Mann-Whitney U Test
u_stat, p_value_mw = mannwhitneyu(correct_confidence, incorrect_confidence_clean, alternative='two-sided')

print(f"Mann-Whitney U statistic: {u_stat}, P-value: {p_value_mw}")

#GPT4.0 U检验
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu

df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

# Calculate mean and standard deviation (SD) of confidence for correct answers
correct_confidence = df[df['correct_gpt4'] == 1]['gpt-4_confidence_answer_likert']
mean_correct_confidence = correct_confidence.mean()
sd_correct_confidence = correct_confidence.std()

# Calculate mean and SD of confidence for incorrect answers
incorrect_confidence = df[df['correct_gpt4'] == 0]['gpt-4_confidence_answer_likert']
mean_incorrect_confidence = incorrect_confidence.mean()
sd_incorrect_confidence = incorrect_confidence.std()

print(f"Correct answers - Mean: {mean_correct_confidence}, SD: {sd_correct_confidence}")
print(f"Incorrect answers - Mean: {mean_incorrect_confidence}, SD: {sd_incorrect_confidence}")

# Remove NaN values from incorrect confidence scores
incorrect_confidence_clean = incorrect_confidence.dropna()
correct_confidence_clean=correct_confidence.dropna()

# Perform the Mann-Whitney U Test
u_stat, p_value_mw = mannwhitneyu(correct_confidence_clean, incorrect_confidence_clean, alternative='two-sided')

print(f"Mann-Whitney U statistic: {u_stat}, P-value: {p_value_mw}")

#%%
#incorrect LLM1 LLM2 confidence
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu

df = pd.read_excel('final_gpt35_gpt4_2.xlsx')
incorrect3_confidence = df[df['correct_gpt3.5'] == 0]['gpt-3.5-turbo_confidence_answer_likert']
incorrect4_confidence = df[df['correct_gpt4'] == 0]['gpt-4_confidence_answer_likert']

proportion3_4_or_5 = (incorrect3_confidence >= 4).sum() / incorrect3_confidence.count()


proportion4_4_or_5 = (incorrect4_confidence >= 4).sum() / incorrect4_confidence.count()


num_confidence3_4_or_5 = (incorrect3_confidence >= 4).sum()
total_incorrect3 = incorrect3_confidence.count()

num_confidence4_4_or_5 = (incorrect4_confidence >= 4).sum()
total_incorrect4 = incorrect4_confidence.count()

print(proportion3_4_or_5,num_confidence3_4_or_5,total_incorrect3)
print(proportion4_4_or_5,num_confidence4_4_or_5,total_incorrect4)


# %%
#repeatability
import pandas as pd

# 加载CSV文件

df = pd.read_csv('df_gpt3_repetitive_letters_50RANDOM.csv')

# 获取所有可能的回答（例如A, B, C, D, E等）
all_answers = set(''.join(df['gpt-3.5-turbo_answer_letters_bound'].tolist()))

# 为每个可能的回答创建一个百分比列
for answer in all_answers:
    # 计算每个回答的出现次数
    df[f'{answer}_repeatability'] = df['gpt-3.5-turbo_answer_letters_bound'].apply(lambda x: x.count(answer) / 50.0)

# 显示更新后的DataFrame
print(df.head())

df.to_csv('df_gpt3_repetitive_letters_50RANDOM.csv', index=False)

#%%
import pandas as pd

# 加载CSV文件

df = pd.read_excel('df_gpt3_repetitive_letters_50RANDOM.xlsx')

# Filtering columns with high repeatability (percentage > 0.75) and creating the 'high_repeatability' column

# Columns to check for high repeatability
repeatability_columns = [col for col in df.columns if col.endswith('_repeatability')]

# Initialize the high_repeatability column as an empty string
df['high_repeat'] = ''

for col in repeatability_columns:
    # For each row, check if the percentage is greater than 0.75
    # If so, append the first letter of the column name (which represents the answer) to the 'high_repeatability' column
    df['high_repeat'] += df.apply(lambda row: col[0] if row[col] > 0.75 else '', axis=1)



df.to_excel('df_gpt3_repetitive_letters_50RANDOM.xlsx', index=False)

import pandas as pd
df = pd.read_excel('df_gpt3_repetitive_letters_50RANDOM.xlsx')


# Identify all the percentage columns again since the state was reset

repeatability_columns = [col for col in df.columns if col.endswith('_repeatability')]


# 定义一个函数来找到当high_repeatability列为空时的最大_repeatability列的首字母
def find_highest_repeatability(row):
    # 如果high_repeatability列为空
    if pd.isna(row['high_repeat']):
        # 初始化一个字典来存储选定的_repeatability列的值
        selected_columns = {col: row[col] for col in repeatability_columns}
        # 找出这些列中最大值的列名
        highest_col = max(selected_columns, key=lambda k: selected_columns[k])
        # 返回该列名的首字母，作为答案
        return highest_col[0]
    # 如果high_repeatability列不为空，则返回空字符串
    return ''

# 应用这个函数到每一行来创建low_repeat列
df['low_repeat'] = df.apply(find_highest_repeatability, axis=1)


# Save the updated DataFrame to a new CSV file to avoid overwriting the original file

df.to_excel('df_gpt3_repetitive_letters_50RANDOM.xlsx', index=False)


#%%
##LLM1高重复性答案与低重复性答案的比较
import pandas as pd

# 假设df是已经加载的DataFrame
df = pd.read_excel('df_gpt3_repetitive_letters_50RANDOM.xlsx')

# 计算high_repeat列与Correct Answer Letter列相等的行数
correct_highmatches = (df['high_repeat'] == df['Correct Answer Letter']).sum()

# 计算high_repeat列非空的行数
non_empty_high_repeat = df['high_repeat'].notnull().sum()

print(f"The number of rows in the high_repeat column that are equal to the Correct Answer Letter column: {correct_highmatches}")
print(f"The number of non-empty rows in the high repeat column: {non_empty_high_repeat}")

# 计算low_repeat列与Correct Answer Letter列相等的行数
correct_lowmatches = (df['low_repeat'] == df['Correct Answer Letter']).sum()

# 计算low_repeat列非空的行数
non_empty_low_repeat = df['low_repeat'].notnull().sum()


print(f"The number of rows in the low_repeat column that are equal to the Correct Answer Letter column: {correct_lowmatches}")
print(f"The number of non-empty rows in the low repeat column: {non_empty_low_repeat}")

from scipy.stats import chi2_contingency


# 创建列联表
# 第一行为每个重复性水平的正确匹配数
# 第二行为每个重复性水平的总非空数减去正确匹配数，即错误匹配数
contingency_table = [
    [correct_lowmatches, correct_highmatches],
    [non_empty_low_repeat - correct_lowmatches, non_empty_high_repeat - correct_highmatches]
]

# 进行卡方检验
chi2, p, dof, ex = chi2_contingency(contingency_table, correction=False)

print(f"Chi-squared: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{ex}")




#%%
import pandas as pd

# 加载CSV文件

df = pd.read_excel('df_gpt4_repetitive_letters_50RANDOM.xlsx')

# 获取所有可能的回答（例如A, B, C, D, E等）
all_answers = set(''.join(df['gpt-4_answer_letters_bound'].tolist()))

# 为每个可能的回答创建一个百分比列
for answer in all_answers:
    # 计算每个回答的出现次数
    df[f'{answer}_repeatability'] = df['gpt-4_answer_letters_bound'].apply(lambda x: x.count(answer) / 50.0)

# 显示更新后的DataFrame
print(df.head())

df.to_excel('df_gpt4_repetitive_letters_50RANDOM.xlsx', index=False)

#%%
import pandas as pd

# 加载CSV文件

df = pd.read_excel('df_gpt4_repetitive_letters_50RANDOM.xlsx')

# Filtering columns with high repeatability (percentage > 0.75) and creating the 'high_repeatability' column

# Columns to check for high repeatability
repeatability_columns = [col for col in df.columns if col.endswith('_repeatability')]

# Initialize the high_repeatability column as an empty string
df['high_repeat'] = ''

for col in repeatability_columns:
    # For each row, check if the percentage is greater than 0.75
    # If so, append the first letter of the column name (which represents the answer) to the 'high_repeatability' column
    df['high_repeat'] += df.apply(lambda row: col[0] if row[col] > 0.75 else '', axis=1)



df.to_excel('df_gpt4_repetitive_letters_50RANDOM.xlsx', index=False)

import pandas as pd
df = pd.read_excel('df_gpt4_repetitive_letters_50RANDOM.xlsx')


# Identify all the percentage columns again since the state was reset

repeatability_columns = [col for col in df.columns if col.endswith('_repeatability')]


# 定义一个函数来找到当high_repeatability列为空时的最大_repeatability列的首字母
def find_highest_repeatability(row):
    # 如果high_repeatability列为空
    if pd.isna(row['high_repeat']):
        # 初始化一个字典来存储选定的_repeatability列的值
        selected_columns = {col: row[col] for col in repeatability_columns}
        # 找出这些列中最大值的列名
        highest_col = max(selected_columns, key=lambda k: selected_columns[k])
        # 返回该列名的首字母，作为答案
        return highest_col[0]
    # 如果high_repeatability列不为空，则返回空字符串
    return ''

# 应用这个函数到每一行来创建low_repeat列
df['low_repeat'] = df.apply(find_highest_repeatability, axis=1)


# Save the updated DataFrame to a new CSV file to avoid overwriting the original file

df.to_excel('df_gpt4_repetitive_letters_50RANDOM.xlsx', index=False)


#%%
##LLM2高重复性答案与低重复性答案的比较
import pandas as pd

# 假设df是已经加载的DataFrame
df = pd.read_excel('df_gpt4_repetitive_letters_50RANDOM.xlsx')

# 计算high_repeat列与Correct Answer Letter列相等的行数
correct_highmatches = (df['high_repeat'] == df['Correct Answer Letter']).sum()

# 计算high_repeat列非空的行数
non_empty_high_repeat = df['high_repeat'].notnull().sum()

print(f"The number of rows in the high_repeat column that are equal to the Correct Answer Letter column: {correct_highmatches}")
print(f"The number of non-empty rows in the high repeat column: {non_empty_high_repeat}")

# 计算low_repeat列与Correct Answer Letter列相等的行数
correct_lowmatches = (df['low_repeat'] == df['Correct Answer Letter']).sum()

# 计算low_repeat列非空的行数
non_empty_low_repeat = df['low_repeat'].notnull().sum()


print(f"The number of rows in the low_repeat column that are equal to the Correct Answer Letter column: {correct_lowmatches}")
print(f"The number of non-empty rows in the low repeat column: {non_empty_low_repeat}")

from scipy.stats import chi2_contingency


# 创建列联表
# 第一行为每个重复性水平的正确匹配数
# 第二行为每个重复性水平的总非空数减去正确匹配数，即错误匹配数
contingency_table = [
    [correct_lowmatches, correct_highmatches],
    [non_empty_low_repeat - correct_lowmatches, non_empty_high_repeat - correct_highmatches]
]

# 进行卡方检验
chi2, p, dof, ex = chi2_contingency(contingency_table, correction=False)

print(f"Chi-squared: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies: \n{ex}")

#%%
#repeatability作图


import matplotlib.pyplot as plt
import seaborn as sns

# 数据
categories = ['High Reproducibility', 'Low Reproducibility']
llm1_counts = [24, 8]  # LLM 1 的高再现性和低再现性的正确答案数
llm1_totals = [32, 18]  # LLM 1 的高再现性和低再现性的总答案数
llm2_counts = [41, 0]  # LLM 2 的高再现性和低再现性的正确答案数
llm2_totals = [50, 0]  # LLM 2 的高再现性和低再现性的总答案数

palette = {1:'#00aaff', 0: '#33cc33'}

bar_width = 0.03

llm1_rates = [llm1_counts[0] / llm1_totals[0] * 100, llm1_counts[1] / llm1_totals[1] * 100 if llm1_totals[1] != 0 else 0]
llm2_rates = [llm2_counts[0] / llm2_totals[0] * 100, 0]  # 对于LLM 2 低再现性，我们直接给0，因为没有数据

# Set the style of the seaborn plots
sns.set_style("whitegrid", {'axes.grid' : False})

# 位置

index = [0, 0.12]
# Start creating the plot
plt.figure(figsize=(12, 8))


# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14  # 设置全局字体大小为10

# 使用指定的颜色画图
fig, ax = plt.subplots()

bar1 = ax.bar(index, llm1_rates, bar_width, label='LLM 1', color=palette[1])
bar2 = ax.bar([i + bar_width for i in index], llm2_rates, bar_width, label='LLM 2', color=palette[0])
# 在每个条形上添加数据标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(round(height, 1)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bar1)
add_labels(bar2)

# 设置图形属性
ax.set_xlabel('Reproducibility')
ax.set_ylabel('Accuracy(%)')
ax.set_title('Accuracy by Reproducibility Type', fontsize=16, fontweight='bold')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(categories)
ax.legend(fontsize=14)  # 设置图例字体大小为8

ax.grid(True, axis='y', linestyle='-', linewidth='0.5', color='#ddd', zorder=0, alpha=0.7) 

# Setting the y-axis maximum value to 100
ax.set_ylim(0, 100)

# 显示图形
plt.tight_layout()
# 保存为PDF格式
plt.savefig('C:/Users/子琦/Desktop/LLM/figure_table/Reproducibility.pdf', format='pdf')
plt.savefig('C:/Users/子琦/Desktop/LLM/figure_table/Reproducibility.svg', format='svg')
plt.show()





#%%
#Question length
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, shapiro



df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

# 将信心数据分为正确答案和错误答案两组
length_correct = df[df['correct_gpt3.5'] == 1]['question_length']
length_incorrect = df[df['correct_gpt4'] == 0]['question_length']

# 对正确答案的回答信心进行Shapiro-Wilk检验
w_statistic, p_value = shapiro(length_correct)
print("Correct answer length - W statistic:", w_statistic, "P:", p_value)

# 对错误答案的回答信心进行Shapiro-Wilk检验
w_statistic, p_value = shapiro(length_incorrect)
print("False answer length -W statistic:", w_statistic, "P:", p_value)


# 对于GPT-3.5模型的正确和错误答案长度进行Mann-Whitney U检验
mwu_results_gpt35 = mannwhitneyu(
    df[df['correct_gpt3.5'] == 1]['question_length'],
    df[df['correct_gpt3.5'] == 0]['question_length'],
    alternative='two-sided'
)

#Question length
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# 增加字体大小设置
plt.rcParams.update({'font.size': 18, 'legend.fontsize': 16})

df = pd.read_excel('final_gpt35_gpt4_2.xlsx')

# 将信心数据分为正确答案和错误答案两组
length_correct = df[df['correct_gpt3.5'] == 1]['question_length']
length_incorrect = df[df['correct_gpt4'] == 0]['question_length']

# 对于GPT-4模型的正确和错误答案长度进行Mann-Whitney U检验
mwu_results_gpt4 = mannwhitneyu(
    df[df['correct_gpt4'] == 1]['question_length'],
    df[df['correct_gpt4'] == 0]['question_length'],
    alternative='two-sided'
)

# 对于GPT-3.5模型的正确和错误答案长度进行Mann-Whitney U检验
mwu_results_gpt35 = mannwhitneyu(
    df[df['correct_gpt3.5'] == 1]['question_length'],
    df[df['correct_gpt3.5'] == 0]['question_length'],
    alternative='two-sided'
)


# 根据提供的条件准备数据
df['Percentage'] = df['Percentage'].str.rstrip('%').astype('float') / 100.0

# Set the style of the seaborn plots
sns.set_theme(style="whitegrid")

# 定义自定义颜色方案
custom_palette = ['#00aaff', '#33cc33']

# 分别为GPT-3.5和GPT-4添加'Correctness'标签
df['Correctness LLM1'] = df['correct_gpt3.5'].map({0: 'Incorrect', 1: 'Correct'})
df['Correctness LLM2'] = df['correct_gpt4'].map({0: 'Incorrect', 1: 'Correct'})

# 数据预处理，这里我们创建一个用于绘图的长格式DataFrame
df_long = pd.melt(df, id_vars=['question_length'], value_vars=['Correctness LLM1', 'Correctness LLM2'],
                  var_name='Model', value_name='Correctness')

# 替换列以匹配模型名称
df_long['Model'] = df_long['Model'].str.replace('Correctness ', '')

# 绘制箱型图
plt.figure(figsize=(6, 8))
ax = sns.boxplot(x='Model', y='question_length', hue='Correctness', data=df_long,
                 palette=custom_palette, width=0.4)


# 设置图表标题和轴标签
ax.set_title('Question Length between LLM 1 and LLM 2', fontsize=16)
ax.set_xlabel('Model', fontsize=16)
ax.set_ylabel('Question Length', fontsize=16)


# Place the p-values; adjust these coordinates based on your data and aesthetics
ax.text(0, max(df_long['question_length']) + 10, f'p={mwu_results_gpt35.pvalue:.3f}', ha='center', va='bottom')
ax.text(1, max(df_long['question_length']) + 10, f'p={mwu_results_gpt4.pvalue:.3f}', ha='center', va='bottom')

# Adjust the legend position

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper center',fontsize=14, bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=True)

# Save the figure to a vector file format (e.g., .pdf)
plt.savefig('C:/Users/子琦/Desktop/LLM/figure_table/question_length_boxplots.pdf', format='pdf')
plt.savefig('C:/Users/子琦/Desktop/LLM/figure_table/question_length_boxplots.svg', format='svg')

plt.show()


#%%
# 散点图
#皮尔逊相关系数计算
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

df = pd.read_excel("final_gpt35_gpt4_2.xlsx")
# 根据提供的条件准备数据
df['Percentage'] = df['Percentage'].str.rstrip('%').astype('float') / 100.0
# 计算实际的相关系数R和p值
r, p_value = pearsonr(df['Percentage'], df['question_length'])

# 显示R值和p值
print(r, p_value)




#%%
#散点图
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
df = pd.read_excel("final_gpt35_gpt4_2.xlsx")

# Set the font to Arial, 10 point for figure text and 8 point for legend
plt.rc('font', family='Arial', size=16)
plt.rc('legend', fontsize=14)

df['Percentage'] = df['Percentage'].str.rstrip('%').astype('float') / 100.0

# Create a custom
custom_colors = ['#34a853', '#4285f4', '#ea4335']
cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)


# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter('Percentage', 'question_length', data=df, c='Percentage', cmap=cmap, alpha=0.6)

# 添加颜色条
cbar = plt.colorbar(scatter, label='Correct (%)', shrink=0.3, aspect=7)
cbar.set_ticks([0.25, 0.5, 0.75, 1])
cbar.set_ticklabels(['25%', '50%', '75%', '100%'])
cbar.ax.tick_params(labelsize=16)

# 添加相关性和p值的注释

ax.text(0.05, 0.95, 'R = -0.004, p = 0.920', transform=ax.transAxes, color='black', fontsize=16, ha='left', va='top')

# 设置标签和标题
ax.set_xlabel('Human Correct (%)')
ax.set_ylabel('Question Length (characters)')
ax.set_title('Human', fontsize=16)






# 将图表保存为PDF格式
fig.savefig('C:/Users/子琦/Desktop/LLM/figure_table/scatter_plot.pdf', format='pdf')
fig.savefig('C:/Users/子琦/Desktop/LLM/figure_table/scatter_plot.svg', format='svg')
plt.show()


#%%
#整体t-SNE可视化图
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns

# 加载嵌入向量数据
# 假设嵌入向量的每行是一个问题的嵌入，并且第一列是索引,分对错
embeddings = pd.read_csv("embeddings_per_question.csv", index_col=0)

# 初始化t-SNE
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)

# 使用t-SNE进行降维
vis_dims = tsne.fit_transform(embeddings)

# 将t-SNE结果保存为CSV文件
pd.DataFrame(vis_dims, index=embeddings.index).to_csv("embeddings_dims.csv")



##分category可视化图
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
import numpy as np


# 加载数据
embeddings_df = pd.read_csv("embeddings_per_question.csv", index_col=0)

# 提取类别信息，假设ID的格式是 'ID_X_Category'
# 这里我们通过分割字符串并获取最后一个元素来提取类别
categories = embeddings_df.index.to_series().apply(lambda x: x.split('_')[-1])

# 初始化t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)

# 使用t-SNE降维，除去第一列的ID信息
tsne_results = tsne.fit_transform(embeddings_df.iloc[:, 1:])

# 创建一个DataFrame来保存t-SNE结果和类别信息
tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
tsne_df['Category'] = categories.values

# Create the TSNE visualization for both the GPT-4 correctness and categories
fig, axes = plt.subplots(2, 1, figsize=(14, 10))  # 2 rows, 1 column

# 设置字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16  # 设置图内文字大小
plt.rcParams['legend.fontsize'] = 14  # 设置图例文字大小



sns.scatterplot(x='TSNE1', y='TSNE2', hue='Category', data=tsne_df, palette=sns.color_palette("hsv", len(tsne_df['Category'].unique())), ax=axes[0])
# 固定图例位置和框大小
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., title='Category')
# 隐藏上框和右框线
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)


# 正确错误可视化
embeddings_dims= pd.read_csv("embeddings_dims.csv", index_col=0)

# Rename the columns correctly for t-SNE dimensions
embeddings_dims.rename(columns={'0': 'TSNE1', '1': 'TSNE2'}, inplace=True)

# Extract the t-SNE dimensions and the 'correct_gpt4' label
vis_dims = embeddings_dims[['TSNE1', 'TSNE2']].values
correct_gpt4 = embeddings_dims['correct_gpt4']

palette = {True: "#4285F4", False: "#DB4437"}
# Plot using seaborn with the corrected palette and labels
scatter = sns.scatterplot(x='TSNE1', y='TSNE2',
                          hue=correct_gpt4, palette=palette,
                          data=embeddings_dims, legend=False, ax=axes[1])  # Turn off automatic legend
# Set titles and labels


# 为第二个子图添加手动图例
legend_labels = ['Incorrect', 'Correct']
legend_colors = [palette[False], palette[True]]
for color, label in zip(legend_colors, legend_labels):
    axes[1].scatter([], [], color=color, label=label)
# 固定图例位置和框大小
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., title='GPT-4')

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Place the legend outside the top right corner of the plot
# 为了防止图例和图表重叠，可以调整子图之间的间距
plt.subplots_adjust(hspace=0.3, right=0.85)
# 显示图表
plt.tight_layout()  # 调整布局以防止重叠
plt.show()

# 将图表保存为PDF格式
fig.savefig('C:/Users/子琦/Desktop/LLM/figure_table/T-SNE_analysis.pdf', format='pdf')
fig.savefig('C:/Users/子琦/Desktop/LLM/figure_table/T-SNE_analysis.svg', format='svg')
plt.show()

#%%
#余弦相似度与正确答案
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
# 读取 CSV 文件
df = pd.read_csv('similarities.csv')

# 找出每个问题中相似性最高的答案
df['Max Similarity Answer'] = df[['A','B','C','D','E','F','G','H','I']].idxmax(axis=1)
# 总问题数量
total_questions = len(df)

# 计算最相似的答案是正确答案的问题数量
correct_count = (df['Max Similarity Answer'] == df['Correct Answer Letter']).sum()
correct_ratio = correct_count / total_questions
print(correct_count, total_questions, correct_ratio)

turbo_count =(df['Max Similarity Answer'] == df['gpt-3.5-turbo_answerletter']).sum()
turbo_count_ratio = turbo_count/ total_questions
print(turbo_count, total_questions, turbo_count_ratio)

forth_count= (df['Max Similarity Answer'] == df['gpt-4_answerletter']).sum()
forth_count_ratio = forth_count/ total_questions
print(forth_count,total_questions,forth_count_ratio)



# 统计数据
count = [forth_count, correct_count]
nobs = [total_questions, total_questions]

# 执行 Z-test for Proportions
stat, pval = proportions_ztest(count, nobs)

print(f'Z-test for GPT-4 vs Correct Count: Z = {stat:.2f}, p-value = {pval:.4f}')

# 重复相同的过程，比较 GPT-3.5-turbo 和 Correct Count
count_turbo = [turbo_count, correct_count]
stat_turbo, pval_turbo = proportions_ztest(count_turbo, nobs)

print(f'Z-test for GPT-3.5-turbo vs Correct Count: Z = {stat_turbo:.2f}, p-value = {pval_turbo:.4f}')

import pandas as pd
print(pd.__version__)
import scipy
import numpy

print("SciPy version:", scipy.__version__)
print("NumPy version:", numpy.__version__)
import statsmodels
print("statsmodels version:", statsmodels.__version__)