from datasources import get_uvas_tseries
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from widgets import error_message


def density_agg_cat(x):
    btypes = {"high-density-types": ["Departamento"], "low-density-types": ["Casa", "PH"]}
    if x in btypes["high-density-types"]:
        return "high-density-types"
    elif x in btypes["low-density-types"]:
        return "low-density-types"
    else:
        return "other"


def land_use_agg_cat(x):
    btypes = {
        "residential-types": ["Casa", "PH", "Departamento"],
        "non-residential-types": ["Oficina", "Depósito", "Local comercial"],
    }
    if x in btypes["residential-types"]:
        return "residential-types"
    elif x in btypes["non-residential-types"]:
        return "non-residential-types"
    else:
        return "other"


def build_project_class(x, target_group, comparison_group):
    # TODO: documentation
    # last class name in alphabetical order is the target class
    # groups = ['comparison_group','target_group']
    # target = groups[-1]
    btypes = {"target-group": target_group, "comparison-group": comparison_group}
    if x in btypes["target-group"]:
        return "target-group"
    elif x in btypes["comparison-group"]:
        return "comparison-group"
    else:
        return "other"


def estimate_uva_pct_growth(
    permission_month: int, permission_year: int, uva_last_avbl_date: pd.Timestamp
) -> float:
    """
    Returns the UVA percent gorwth from 2019 up to
    the construction permission date
    """

    uva_historic_vals = get_uvas_tseries()
    uva_first_avbl_value = uva_historic_vals.head(1)["value"].item()
    uva_last_avbl_value = uva_historic_vals.tail(1)["value"].item()

    permission_month_str = str(permission_month)
    permission_year_str = str(permission_year)

    if len(permission_month_str) < 2:
        permission_month_str = "0" + permission_month_str

    permission_date_str = f"{permission_month_str}/25/{permission_year_str[2:]}"
    permission_datetime = datetime.strptime(permission_date_str, "%m/%d/%y")

    if permission_datetime < uva_last_avbl_date:
        timelimit = str(uva_last_avbl_date.year) + "-" + str(uva_last_avbl_date.month)
        error_message(f"Permission date must be greater than {timelimit}")

    cumsum_period = permission_datetime - uva_last_avbl_date
    diff_months = int(cumsum_period.days / 30)

    past_date = uva_last_avbl_date - relativedelta(months=diff_months)

    uva_historic_cumsum = (
        uva_historic_vals.loc[uva_historic_vals.date >= past_date, "value"].pct_change().sum()
    )

    forecast_end_date_value = uva_last_avbl_value + uva_last_avbl_value * uva_historic_cumsum
    # st.write(uva_first_avbl_value)
    # st.write(forecast_end_date_value)

    total_growth = forecast_end_date_value / uva_first_avbl_value - 1

    return total_growth


def update_urban_land_incidence(parcel_uva: float, pct_growth: float) -> float:
    """
    Updates the urban land incidence value of parcels
    adding the UVA percentage growth from 2019 up to the permission date
    """
    adjusted_value = parcel_uva + parcel_uva * pct_growth
    return adjusted_value
