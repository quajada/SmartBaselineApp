# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:24:50 2022

@author: Bruno Tabet
"""

from streamlit.components.v1 import html
import pandas as pd
import numpy as np

def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date() - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)
    
    
def initialization(st):
    
    if 'home' not in st.session_state:
        st.session_state['home'] = 0
    
    if 'iter' not in st.session_state:
        st.session_state['iter'] = 0
    
    if 'file_uploaded' not in st.session_state:
        st.session_state['file_uploaded'] = 0
    
    if 'data_uploaded' not in st.session_state:
        st.session_state['data_uploaded'] = 0
    
    if 'outliers_removal' not in st.session_state:
        st.session_state['outliers_removal'] = 0
    
    if 'synthetic_features_created' not in st.session_state:
        st.session_state['synthetic_features_created'] = 0
    
    if 'filters_applied' not in st.session_state:
        st.session_state['filters_applied'] = 0
    
    if 'filters_manual' not in st.session_state:
        st.session_state['filters_manual'] = 0
    
    if 'regression_done' not in st.session_state:
        st.session_state['regression_done'] = 0
    
    if 'results' not in st.session_state:
        st.session_state['results'] = 0
        
    if 'database' not in st.session_state:
        st.session_state['database'] = 0
    
    if 'feature_names' not in st.session_state:
        st.session_state['feature_names'] = {}
        
    if 'selected_pt' not in st.session_state:
        st.session_state['selected_pt'] = {}
        
    # if 'selected_points' not in st.session_state:
    #     st.session_state['selected_points'] = {}
        
    if 'outliers_points' not in st.session_state:
        st.session_state['outliers_points'] = {}
        
    if 'selected_points2' not in st.session_state:
        st.session_state['selected_points2'] = {}
    
    if 'outliers_points2' not in st.session_state:
        st.session_state['outliers_points2'] = {}
    
    if 'button' not in st.session_state:
        st.session_state['button'] = False
    
    if 'first' not in st.session_state:
        st.session_state['first'] = True
    
    if 'final_path' not in st.session_state:
        st.session_state['final_path'] = 0
    
    if 'results_df_outliers' not in st.session_state:
        st.session_state['results_df_outliers'] = pd.DataFrame(columns = ['combinations', 'version', 'r2', 'std_dev', 'r2_cv_test', 'std_dev_cv_test', 'intercept', 'pval', 'tval', 'cv_rmse', 'IPMVP_compliant', 'AIC', 'AIC_adj', 'size', 'nb_data_points', 'nb_outliers_removed'])
    
    if 'tab_seleted' not in st.session_state:
        st.session_state['tab_selected'] = 'Home'
    
    if 'tabs' not in st.session_state:
        st.session_state['tabs'] = ['Home', 'Dataset', 'Synthetic features', 'Filters', 'Regression', 'Results', 'Database']
    
    if 'Next_key_Dataset' not in st.session_state:
        st.session_state['Next_key_Dataset'] = False
        
    if 'M&V' not in st.session_state:
        st.session_state['M&V'] = 1

    st.set_page_config(
        layout='wide',
    )


def initialize_selections(st):
    
    if 'results_dict' in st.session_state:
        for combination in st.session_state['results_dict']:
            st.session_state['iteration' + str(combination)] = 1
            
            
    
def initialize_selected_outliers_points(st):
    st.session_state['selected_points'] = {}
    st.session_state['outliers_points'] = {}
    for x in st.session_state['y_df_dataset'].index:
        st.session_state['selected_points'][x]= {}
        st.session_state['selected_points'][x]['date'] = st.session_state['y_df_with_dates']['From (incl)'][x]
        st.session_state['selected_points'][x]['y'] = st.session_state['y_df_with_dates']['Normalized baseline'][x]
    

def initialize_results_df_outliers(st):
    st.session_state['results_df_outliers'] = pd.DataFrame(columns = ['combinations', 'version', 'r2', 'std_dev', 'r2_cv_test', 'std_dev_cv_test', 'intercept', 'pval', 'tval', 'cv_rmse', 'IPMVP_compliant', 'AIC', 'AIC_adj', 'size', 'nb_data_points', 'nb_outliers_removed'])




def delete_selected(st):
    for key in st.session_state:
        if type(key) == str:
            if 'selected_points(' in key:
                del st.session_state[key]
            if 'outliers_points(' in key:
                del st.session_state[key]

def display_results(sel_combi2, sel_version, st):
    
    r2 = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['r2']
    std_dev = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['std_dev']
    r2_cv_test = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['r2_cv_test']
    std_dev_cv_test = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['std_dev_cv_test']
    cv_rmse = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['cv_rmse']
    AIC = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['AIC']
    AIC_adj = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['AIC_adj']
    tval = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['tval']
    pval = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['pval']
    y_pred = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['y_pred']
    
    
    coefs = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['coefs']
    
    coefsUSA = [0]*len(coefs)
    
    for i in range(len(coefs)):
        
        coefsUSA[i] = float('%.5g' % coefs[i])
        coefsUSA[i] = np.format_float_scientific(np.float(coefsUSA[i]))
        # coefsUSA[i] = round(coefs[i], 4)
        # coefsUSA[i] = "{:,}".format(coefsUSA[i])
    
    if coefs[0] < 0:
        intercept_ = '**'
    else:
        intercept_ = ''


    name = 'The feature'
    if len(sel_combi2) == 1:
        name += ' is '+ str(sel_combi2[0])
    else:
        name += 's are ' + str(sel_combi2[0])
        for i in range (1, len(sel_combi2)):
            name += ' and ' + str(sel_combi2[i])
    name += '.'
    
    equation = 'normalized baseline = '+ str(coefsUSA[0]) + ' '
    
    for i in range (len(sel_combi2)):
        if coefs[i+1] < 0:    
            equation += str(coefsUSA[i+1]) + ' * ' + str(sel_combi2[i])
        else:
            equation += ' + ' + str(coefsUSA[i+1]) + ' * ' + str(sel_combi2[i])

    
    st.subheader("The regression equation is :")
    st.subheader(equation)
    
    st.write(name)
    st.write("Here are the regression statistics and parameters of the model you've selected " +
             "(the statistics and parameters in bold are the ones that could be problematic).")

    st.session_state['equation'] = equation
    
    st.write(intercept_ + 'intercept = ' + coefsUSA[0] + intercept_)
    
    for i in  range(len(sel_combi2)):
        if coefs[i+1] < 0:
            slope_i_ = "**"
        else:
            slope_i_ = ''
        st.write(slope_i_ + 'slope of ' + sel_combi2[i] + ' = ' + coefsUSA[i+1] + slope_i_) 
    
    
    if r2 < 0.75:
        r2_ = '**'
    else:
        r2_ = ''    
    st.write(r2_ + 'r2 = ' + str(r2) + r2_)
    
    if cv_rmse > 0.2:
        cv_rmse_ = '**'
    else:
        cv_rmse_ = ''
    st.write(cv_rmse_ + 'cv_rmse = ' + str(cv_rmse) + cv_rmse_)
    
    st.write('r2_cv_test = ' + str(r2_cv_test))
    
    st.write('std_dev_cv_test = ' + str(std_dev_cv_test))
    
    tval2 = [abs(tval[i]) for i in range (len(tval))]
    if min(tval2) < 2:
        tval_ = '**'
    else:
        tval_ = '' 
    st.write(tval_ + 'tval = ' + str(tval) + tval_)
    
    if max(pval) > 0.1:
        pval_ = '**'
    else:
        pval_ = '' 
    st.write(pval_ + 'pval = '+ str(pval) + pval_)
    

    
    st.write('std_dev = ' + str(std_dev))

    st.write('AIC = ' + str(AIC))
    st.write('AIC_adj = '+ str(AIC_adj))

