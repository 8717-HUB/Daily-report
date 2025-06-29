import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

class TrafficReportGenerator:
    def __init__(self, file_path, sheet1_name='Sheet1', sheet2_name='Sheet2'):
        self.file_path = file_path
        self.sheet1_name = sheet1_name
        self.sheet2_name = sheet2_name
        self.general_summary = None
        self.overloads_data = None
        self.all_stations = [
            "AHERO-KERICHO", "AHERO-KISUMU", "ELDORET-ELDORET", "ELDORET-WEBUYE",
            "MAKUTANO-KAPENGURIA", "MAKUTANO-KITALE", "MALABA-MALABA", "MALABA-UGANDA",
            "MAYONI-BUNGOMA", "MAYONI-MUMIAS", "NGERIA-ELDORET", "NGERIA-NAKURU",
            "SOUTHERN BYPASS-MOMBASA", "SOUTHERN BYPASS-KIKUYU", "ELDAMA RAVINE-ELDAMA RAVINE",
            "ELDAMA RAVINE-ESAGERI", "MUKUMU-CHAVAKALI", "MUKUMU-KAKAMEGA",
            "RONGO-RONGO", "RONGO-SUNEKA", "SALGAA-NAKURU", "SALGAA-TIMBOROA",
            "MADOGO-BANGALI", "MADOGO-GARISSA", "MALINDI-MALINDI", "MALINDI-MAMBRUI",
            "EMALI-EMALI", "EMALI-OLOITOKTOK", "KAJIADO-BISIL", "KAJIADO-ISINYA",
            "KIBWEZI-KIBWEZI", "KIBWEZI-KITUI", "MALILI-MOMBASA", "MALILI-NAIROBI",
            "MWATATE-MWATATE", "MWATATE-VOI", "YATTA-MATUU", "YATTA-THIKA",
            "KAMULU-KAMULU", "KAMULU-TALA", "KALOLENI-KILIFI", "KALOLENI-MARIAKANI",
            "LAISAMIS-MARSABIT", "LAISAMIS-ISIOLO", "SAGANA-NAIROBI", "SAGANA-NYERI"
        ]
        # Set global Seaborn style for professional charts
        sns.set_style("whitegrid", {
            'axes.grid': True,
            'grid.color': '#D3D3D3',
            'grid.linestyle': '--',
            'axes.edgecolor': '#333333',
            'axes.facecolor': 'white',
            'font.family': 'Arial'
        })
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.titlepad': 20,
            'axes.labelpad': 10
        })

    def normalize_station_name(self, name):
        """Normalize station names for accurate comparison"""
        if pd.isna(name):
            return ""
        name = str(name).upper().strip()
        name = re.sub(r'\s+', ' ', name)
        name = re.sub(r'\s*-\s*', '-', name)
        return name

    def wrangle_general_summary(self):
        """Wrangle data from Sheet1 and return General Summary DataFrame"""
        columns = [
            'Station', 'Date', 'Unknown', 'Passengers car',
            '2-axle trucks', '3-axle trucks', '4-axle trucks', '5-axle trucks',
            '6-axle trucks', '7-axle trucks', '8-axle trucks', '2-axle buses',
            '3-axle buses', 'Total buses', 'Axle Vehicle Weight Overload',
            'Gross Vehicle Weight Overload>0<2000', 'Gross Vehicle Weight Overload<2000',
            'No gross vehicle Weight Overloads', 'Gross Vehicle Weight Compliance(%)',
            'Total Traffic'
        ]
        df = pd.read_excel(self.file_path, sheet_name=self.sheet1_name, header=None, usecols=range(1, len(columns)+1))
        if len(df.columns) != len(columns):
            raise ValueError(f"Expected {len(columns)} columns but found {len(df.columns)}")
        df.columns = columns
        numerical_cols = [col for col in df.columns if col not in ['Station', 'Date', 'Gross Vehicle Weight Compliance(%)']]
        totals = df[numerical_cols].sum()
        compliance_avg = df['Gross Vehicle Weight Compliance(%)'].mean()
        totals_row = pd.DataFrame([{
            'Station': 'Totals',
            'Date': '',
            **{col: totals[col] for col in numerical_cols},
            'Gross Vehicle Weight Compliance(%)': compliance_avg
        }])
        df = df.drop(columns="Date").fillna(0)
        df = pd.concat([df, totals_row], ignore_index=True)
        self.general_summary = df
        return df

    def wrangle_overloads(self):
        """Wrangle data from Sheet2 and return Overloads DataFrame"""
        df = pd.read_excel(self.file_path, sheet_name=self.sheet2_name)
        self.overloads_data = df
        return df

    def generate_traffic_summary(self):
        """Generate traffic summary DataFrame and text"""
        if self.general_summary is None:
            self.wrangle_general_summary()
        
        df = self.general_summary.iloc[:-1] if 'Totals' in self.general_summary['Station'].values else self.general_summary
        traffic_summary = pd.DataFrame({
            'Number of vehicles': ['UNKNOWN', 'PASSENGER CARS', 'TRUCKS', 'BUSES', 'TOTAL TRAFFIC']
        })
        traffic_summary['Count'] = [
            int(df['Unknown'].sum()),
            int(df['Passengers car'].sum()),
            int(df[['2-axle trucks', '3-axle trucks', '4-axle trucks', '5-axle trucks',
                   '6-axle trucks', '7-axle trucks', '8-axle trucks']].sum().sum()),
            int(df[['2-axle buses', '3-axle buses']].sum().sum()),
            int(df['Total Traffic'].sum())
        ]
        total_traffic = traffic_summary.loc[4, 'Count']
        traffic_summary['Count relative to total traffic (%)'] = [
            round((x/total_traffic*100), 2) if total_traffic > 0 else 0.0 for x in traffic_summary['Count']
        ]
        traffic_summary['Count relative to total traffic (%)'] = traffic_summary['Count relative to total traffic (%)'].apply(lambda x: f"{x:.2f}%")
        
        total_vehicles = traffic_summary.loc[4, 'Count']
        passenger_cars = traffic_summary.loc[1, 'Count']
        trucks = traffic_summary.loc[2, 'Count']
        unknown = traffic_summary.loc[0, 'Count']
        buses = traffic_summary.loc[3, 'Count']
        
        passenger_pct = round((passenger_cars/total_vehicles*100), 2) if total_vehicles > 0 else 0.0
        trucks_pct = round((trucks/total_vehicles*100), 2) if total_vehicles > 0 else 0.0
        unknown_pct = round((unknown/total_vehicles*100), 2) if total_vehicles > 0 else 0.0
        buses_pct = round((buses/total_vehicles*100), 2) if total_vehicles > 0 else 0.0
        
        summary_text = f"""
Total Traffic Summary
- Total number of vehicles weighed across the stations: {total_vehicles:,.0f}
- Passenger cars: {passenger_cars:,.0f} ({passenger_pct:.2f}%)
- Trucks: {trucks:,.0f} ({trucks_pct:.2f}%)
- Total number of unknown vehicles: {unknown:,.0f} ({unknown_pct:.2f}%)
- Buses: {buses:,.0f} ({buses_pct:.2f}%)
"""
        return traffic_summary, summary_text

    def generate_overloads_summary(self):
        """Generate overloads summary DataFrame and text"""
        if self.general_summary is None:
            self.wrangle_general_summary()
        
        df = self.general_summary.iloc[:-1] if 'Totals' in self.general_summary['Station'].values else self.general_summary
        overloads_summary = pd.DataFrame({
            'Category': [
                'Gross Vehicle Weight Overload >0 <2000',
                'Gross Vehicle Weight Overload >=2000',
                'No Gross Vehicle Weight Overload',
                'TOTAL TRAFFIC'
            ],
            'Number of Vehicles': [
                int(df['Gross Vehicle Weight Overload>0<2000'].sum()),
                int(df['Gross Vehicle Weight Overload<2000'].sum()),
                int(df['No gross vehicle Weight Overloads'].sum()),
                int(df['Total Traffic'].sum())
            ]
        })
        total_traffic = overloads_summary.loc[3, 'Number of Vehicles']
        overloads_summary['Percentage (%)'] = [
            round((x/total_traffic*100), 2) if total_traffic > 0 else 0.0 for x in overloads_summary['Number of Vehicles']
        ]
        overloads_summary['Percentage (%)'] = overloads_summary['Percentage (%)'].apply(lambda x: f"{x:.2f}%")
        
        overload_text = f"""
1.2 Overloads Summary
- Gross vehicle weight overload between 0 and 2,000 kg: {overloads_summary.loc[0, 'Number of Vehicles']:,.0f} ({overloads_summary.loc[0, 'Percentage (%)']})
- Tags - Gross vehicles weight exceeding 2,000 kg: {overloads_summary.loc[1, 'Number of Vehicles']:,.0f} ({overloads_summary.loc[1, 'Percentage (%)']})
- Compliance - Number of vehicles with no overload: {overloads_summary.loc[2, 'Number of Vehicles']:,.0f} ({overloads_summary.loc[2, 'Percentage (%)']})
"""
        return overloads_summary, overload_text

    def generate_missing_stations_report(self):
        """Generate report of missing stations"""
        if self.general_summary is None:
            self.wrangle_general_summary()
        
        normalized_ref = [self.normalize_station_name(s) for s in self.all_stations]
        normalized_gs = [self.normalize_station_name(s) for s in self.general_summary['Station']]
        missing_stations = [self.all_stations[normalized_ref.index(ref_station)]
                           for ref_station in normalized_ref if ref_station not in normalized_gs]
        
        simplified_names = []
        seen = set()
        for station in missing_stations:
            simple_name = station.split('-')[0]
            if simple_name not in seen:
                simplified_names.append(simple_name)
                seen.add(simple_name)
        
        report_lines = ["The following stations did not stream in data:"]
        for i, name in enumerate(simplified_names, 1):
            formatted_name = name.title() if not name.isupper() else ' '.join(word.title() for word in name.split())
            report_lines.append(f"- {formatted_name}")
        
        return '\n'.join(report_lines)

    def generate_station_traffic_analysis(self):
        """Generate station traffic analysis with enhanced bar chart and text"""
        if self.general_summary is None:
            self.wrangle_general_summary()
        
        df = self.general_summary[self.general_summary['Station'] != 'Totals'].copy() if 'Totals' in self.general_summary['Station'].values else self.general_summary.copy()
        df['Station'] = df['Station'].str.split('-').str[0].str.strip().str.title()
        station_traffic = df.groupby('Station', as_index=False)['Total Traffic'].sum().sort_values('Total Traffic', ascending=False)
        
        # Create enhanced bar chart
        fig = plt.figure(figsize=(10, 6))
        palette = sns.color_palette("Blues_r", len(station_traffic))
        ax = sns.barplot(
            data=station_traffic,
            x='Total Traffic',
            y='Station',
            palette=palette,
            edgecolor='#333333',
            linewidth=0.8
        )
        plt.title('Total Traffic per Station', fontsize=16, pad=20, weight='bold')
        plt.xlabel('Number of Vehicles', fontsize=14)
        plt.ylabel('Station', fontsize=14)
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,.0f}"))
        ax.tick_params(axis='y', labelsize=10)
        
        # Add value labels with shadow for readability
        for p in ax.patches:
            width = p.get_width()
            ax.text(
                width + (station_traffic['Total Traffic'].max() * 0.02),
                p.get_y() + p.get_height()/2,
                f"{int(width):,.0f}",
                va='center',
                ha='left',
                fontsize=10,
                color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
            )
        
        plt.tight_layout()
        
        station_summary = pd.DataFrame({
            'STATION': station_traffic['Station'],
            'TRAFFIC': station_traffic['Total Traffic'].apply(lambda x: f"{int(x):,.0f}"),
            'PERCENTAGE': [
                round((x/station_traffic['Total Traffic'].sum()*100), 2) if station_traffic['Total Traffic'].sum() > 0 else 0.0
                for x in station_traffic['Total Traffic']
            ]
        })
        station_summary['PERCENTAGE'] = station_summary['PERCENTAGE'].apply(lambda x: f"{x:.2f}%")
        
        total_traffic = station_traffic['Total Traffic'].sum()
        top_stations = station_traffic.nlargest(3, 'Total Traffic')
        bottom_station = station_traffic.nsmallest(1, 'Total Traffic')
        mid_range = station_traffic[~station_traffic['Station'].isin(top_stations['Station'])].copy()
        mid_range_stats = mid_range['Total Traffic'].agg(['min', 'max'])
        
        analysis_text = f"""
- {top_stations.iloc[0]['Station']} had the highest traffic volume, with {int(top_stations.iloc[0]['Total Traffic']):,.0f} vehicles, representing {round((top_stations.iloc[0]['Total Traffic']/total_traffic*100), 2):.2f}% of the total traffic.
- Other stations with significant traffic included {top_stations.iloc[1]['Station']} ({int(top_stations.iloc[1]['Total Traffic']):,.0f}) and {top_stations.iloc[2]['Station']} ({int(top_stations.iloc[2]['Total Traffic']):,.0f}).
- The rest of the stations recorded vehicle counts varying from {int(mid_range_stats['min']):,.0f} to {int(mid_range_stats['max']):,.0f}, with {bottom_station.iloc[0]['Station']} recording the lowest traffic.

Additional Insights:
- The top 3 stations accounted for {round((top_stations['Total Traffic'].sum()/total_traffic*100), 2):.2f}% of total traffic, indicating high concentration at key locations.
- Traffic distribution shows {len(station_traffic[station_traffic['Total Traffic'] > total_traffic/len(station_traffic)])} stations above average traffic and {len(station_traffic[station_traffic['Total Traffic'] < total_traffic/len(station_traffic)])} below average.
- The median station traffic was {int(station_traffic['Total Traffic'].median()):,.0f} vehicles, suggesting many stations handle moderate volumes.
"""
        return fig, station_summary, analysis_text

    def generate_trucks_distribution(self):
        """Generate trucks distribution analysis with enhanced donut chart, table, and text"""
        if self.general_summary is None:
            self.wrangle_general_summary()
        
        df = self.general_summary.iloc[:-1] if 'Totals' in self.general_summary['Station'].values else self.general_summary.copy()
        
        truck_categories = {
            '2 Axle': '2-axle trucks', '3 Axle': '3-axle trucks', '4 Axle': '4-axle trucks',
            '5 Axle': '5-axle trucks', '6 Axle': '6-axle trucks', '7 Axle': '7-axle trucks',
            '8 Axle': '8-axle trucks'
        }
        counts = [int(df[col].sum()) for col in truck_categories.values()]
        total_trucks = sum(counts)
        percentages = [round((x/total_trucks*100), 2) if total_trucks > 0 else 0.0 for x in counts]
        
        average_axles = sum([(i+2)*count for i, count in enumerate(counts)])/total_trucks if total_trucks > 0 else 0.0
        heavy_truck_pct = sum(percentages[3:7])
        top_config_pct = percentages[0] + percentages[4]
        
        analysis_text = f"""
- Two-axle trucks comprised a substantial portion of the total truck traffic, totaling {counts[0]:,.0f} vehicles, or {percentages[0]:,.2f}%. This indicates that most trucks on the road feature a simpler axle configuration, likely reflecting the common use of standard freight and transport vehicles.

- Six-axle trucks accounted for {percentages[4]:,.2f}% of the truck traffic, with {counts[4]:,.0f} vehicles recorded. This significant share points to the notable presence of larger, heavier trucks that play a vital role in carrying substantial loads and may contribute to road wear.

- Five-axle ({percentages[3]:,.2f}%) and Three-axle trucks ({percentages[1]:,.2f}%) represent smaller segments of truck traffic, indicating configurations that are less common yet still important.
- Four-axle trucks ({percentages[2]:,.2f}%) and seven-axle trucks ({percentages[5]:,.2f}%) were even rarer, with {'no' if counts[6] == 0 else f'{counts[6]:,.0f}'} instance of eight-axle truck reported.

Additional Insights:
- The top 2 configurations (2-axle and 6-axle) account for {top_config_pct:.2f}% of all truck traffic, showing high concentration.
- Heavy trucks (5+ axles) represent {heavy_truck_pct:.2f}% of truck traffic, important for infrastructure planning.
- The average number of axles per truck is approximately {average_axles:.2f}, indicating typical vehicle size.
"""
        # Create enhanced donut chart
        fig = plt.figure(figsize=(8, 6))
        palette = sns.color_palette("Set2", len(truck_categories))
        wedges, texts, autotexts = plt.pie(
            counts,
            labels=[f"{label} ({count:,.0f})" for label, count in zip(truck_categories.keys(), counts)],
            autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
            startangle=90,
            colors=palette,
            wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1),
            textprops={'fontsize': 10, 'weight': 'bold'},
            pctdistance=0.85
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        plt.gca().add_artist(centre_circle)
        plt.title('Truck Traffic Distribution by Axle Configuration', fontsize=16, pad=20, weight='bold')
        plt.legend(
            wedges, 
            [f"{label} ({count:,.0f}, {pct:.1f}%)" for label, count, pct in zip(truck_categories.keys(), counts, percentages)],
            title="Axle Configurations",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=10,
            title_fontsize=12
        )
        plt.axis('equal')
        plt.tight_layout()
        
        trucks_distribution = pd.DataFrame({
            'Axle Configuration': list(truck_categories.keys()) + ['TOTAL TRUCKS'],
            'Total Count': [f"{x:,.0f}" for x in counts] + [f"{total_trucks:,.0f}"],
            'Percentage': [f"{x:.2f}%" for x in percentages] + ["100.00%"]
        })
        
        return fig, trucks_distribution, analysis_text

    def generate_vehicle_category_analysis(self):
        """Generate vehicle category analysis with enhanced pie and bar charts, table, and text"""
        if self.general_summary is None:
            self.wrangle_general_summary()
        
        df = self.general_summary.iloc[:-1] if 'Totals' in self.general_summary['Station'].values else self.general_summary.copy()
        
        categories = {
            'Passenger cars': int(df['Passengers car'].sum()),
            '2-axle trucks': int(df['2-axle trucks'].sum()),
            'Unknown': int(df['Unknown'].sum()),
            '6-axle trucks': int(df['6-axle trucks'].sum()),
            '3-axle trucks': int(df['3-axle trucks'].sum()),
            '5-axle trucks': int(df['5-axle trucks'].sum()),
            '2-axle buses': int(df['2-axle buses'].sum()),
            '3-axle buses': int(df['3-axle buses'].sum()),
            '4-axle trucks': int(df['4-axle trucks'].sum()),
            '7-axle trucks': int(df['7-axle trucks'].sum()),
            '8-axle trucks': int(df['8-axle trucks'].sum())
        }
        total_traffic = int(df['Total Traffic'].sum())
        percentages = {k: round((v/total_traffic*100), 2) if total_traffic > 0 else 0.0 for k, v in categories.items()}
        
        top_category = max(categories, key=categories.get)
        top_percentage = percentages[top_category]
        truck_categories = [k for k in categories if 'truck' in k.lower()]
        total_trucks = sum(categories[k] for k in truck_categories)
        truck_percentage = round((total_trucks/total_traffic*100), 2) if total_traffic > 0 else 0.0
        bus_categories = [k for k in categories if 'bus' in k.lower()]
        total_buses = sum(categories[k] for k in bus_categories)
        bus_percentage = round((total_buses/total_traffic*100), 2) if total_traffic > 0 else 0.0
        unknown_percentage = percentages['Unknown']
        
        analysis_text = f"""
- The data on vehicle distribution indicates that {top_category} were the most common category, making up {top_percentage:.2f}% of the total traffic.
- 2-axle trucks accounted for {percentages['2-axle trucks']:.2f}%, Unknown comprised a notable {unknown_percentage:.2f}% while 6-axle trucks represented {percentages['6-axle trucks']:.2f}% of the total traffic.

Additional Insights:
- Trucks collectively account for {truck_percentage:.2f}% of total traffic, indicating they dominate the vehicle mix.
- Buses represent only {bus_percentage:.2f}% of traffic, suggesting limited public transport usage.
- The unknown category at {unknown_percentage:.2f}% highlights potential data quality issues worth investigating.
- The top 3 vehicle categories account for {sum(sorted(percentages.values(), reverse=True)[:3]):.2f}% of traffic, showing high concentration.
- Passenger vehicles (cars + buses) make up {(percentages['Passenger cars'] + bus_percentage):.2f}% of traffic versus {truck_percentage:.2f}% for commercial vehicles.
"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        palette = sns.color_palette("Set3", len(categories))
        
        # Pie chart for vehicle groups
        vehicle_groups = {
            'Passenger Vehicles': ['Passenger cars', '2-axle buses', '3-axle buses'],
            'Trucks': [k for k in categories if 'truck' in k.lower()],
            'Unknown': ['Unknown']
        }
        group_data = []
        for group_name, group_items in vehicle_groups.items():
            group_count = sum(categories[item] for item in group_items)
            group_data.append({
                'Category': group_name,
                'Count': group_count,
                'Percentage': round((group_count/total_traffic*100), 2) if total_traffic > 0 else 0.0
            })
        
        group_df = pd.DataFrame(group_data).sort_values('Count', ascending=False)
        
        wedges, texts, autotexts = ax1.pie(
            group_df['Count'],
            labels=group_df['Category'],
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2", 3),
            wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1),
            textprops={'fontsize': 10, 'weight': 'bold'},
            pctdistance=0.85
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax1.add_artist(centre_circle)
        ax1.set_title('Vehicle Group Distribution', fontsize=14, pad=20, weight='bold')
        ax1.legend(
            wedges, 
            [f"{row['Category']} ({row['Count']:,.0f}, {row['Percentage']:.1f}%)" for _, row in group_df.iterrows()],
            title="Vehicle Groups",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=10,
            title_fontsize=12
        )
        ax1.axis('equal')
        
        # Bar chart for detailed categories
        viz_data = pd.DataFrame({
            'Category': list(categories.keys()),
            'Count': list(categories.values()),
            'Percentage': list(percentages.values())
        }).sort_values('Count', ascending=False)
        
        sns.barplot(
            x='Percentage',
            y='Category',
            data=viz_data,
            palette=palette,
            ax=ax2,
            edgecolor='#333333',
            linewidth=0.8
        )
        ax2.set_title('Vehicle Category Breakdown', fontsize=14, pad=20, weight='bold')
        ax2.set_xlabel('Percentage of Total Traffic (%)', fontsize=12)
        ax2.set_ylabel('')
        ax2.tick_params(axis='y', labelsize=10)
        
        for i, (count, pct) in enumerate(zip(viz_data['Count'], viz_data['Percentage'])):
            ax2.text(
                pct + 1,
                i,
                f"{count:,.0f} ({pct:.1f}%)",
                va='center',
                ha='left',
                fontsize=10,
                color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
            )
        
        ax2.set_xlim(0, max(viz_data['Percentage']) * 1.2)
        plt.tight_layout()
        
        vehicle_distribution = pd.DataFrame({
            'Vehicle Category': list(categories.keys()),
            'Total Count': [f"{v:,.0f}" for v in categories.values()],
            'Percentage': [f"{v:.2f}%" for v in percentages.values()]
        }).sort_values('Percentage', ascending=False)
        
        return fig, vehicle_distribution, analysis_text

    def generate_unknowns_analysis(self):
        """Generate unknowns analysis with enhanced scatter plot, table, and text"""
        if self.general_summary is None:
            self.wrangle_general_summary()
        
        df = self.general_summary[self.general_summary['Station'] != 'Totals'].copy() if 'Totals' in self.general_summary['Station'].values else self.general_summary.copy()
        
        df['Unknown_Percentage'] = np.where(
            df['Total Traffic'] > 0,
            ((df['Unknown'] / df['Total Traffic']) * 100).round(2),
            0.0
        ).astype(float)
        
        df = df.sort_values(['Unknown_Percentage', 'Total Traffic'], ascending=[False, False])
        
        top_unknowns = df.nlargest(3, 'Unknown_Percentage')
        low_unknowns = df.nsmallest(4, 'Unknown_Percentage')
        
        # Create enhanced scatter plot
        fig = plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(
            data=df,
            x='Total Traffic',
            y='Unknown_Percentage',
            size='Total Traffic',
            sizes=(100, 1200),
            color='#1f77b4',
            alpha=0.8,
            edgecolor='#333333',
            linewidth=0.8
        )
        ax.axhline(y=20, color='#d62728', linestyle='--', linewidth=2, label='20% Quality Threshold')
        ax.text(
            x=df['Total Traffic'].max() * 0.9,
            y=22,
            s='20% Quality Threshold',
            color='#d62728',
            fontsize=10,
            va='center',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#333333', boxstyle='round,pad=0.3')
        )
        for _, row in df.iterrows():
            va = 'center'
            y_offset = 0
            x_offset = row['Total Traffic'] * 0.03
            if row['Unknown_Percentage'] > 80:
                va = 'bottom'
                y_offset = -2
            elif row['Unknown_Percentage'] < 20:
                va = 'top'
                y_offset = 2
            ax.text(
                x=row['Total Traffic'] + x_offset,
                y=row['Unknown_Percentage'] + y_offset,
                s=row['Station'].split('-')[0].title(),
                fontsize=9,
                ha='left',
                va=va,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3')
            )
        plt.title('Traffic Volume vs Unknown Vehicle Percentage', fontsize=16, pad=20, weight='bold')
        plt.xlabel('Total Traffic (Vehicles)', fontsize=14)
        plt.ylabel('Unknown Vehicles (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 110)
        plt.yticks(range(0, 101, 10))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 10000 else f"{int(x):,.0f}"))
        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        
        analysis_text = f"""
- This section examines the distribution of unknown vehicles across stations. The data reveals variability in the percentage of unknown vehicles, highlighting discrepancies influenced by location, traffic patterns, or route characteristics.

- High unknown percentages may indicate issues with ANPR cameras or vehicle types evading detection.

- {top_unknowns.iloc[0]['Station']} ({top_unknowns.iloc[0]['Unknown_Percentage']:.2f}%), {top_unknowns.iloc[1]['Station']} ({top_unknowns.iloc[1]['Unknown_Percentage']:.2f}%), and {top_unknowns.iloc[2]['Station']} ({top_unknowns.iloc[2]['Unknown_Percentage']:.2f}%) had high unknown ratios, suggesting potential ANPR camera issues.

- Conversely, {low_unknowns.iloc[0]['Station']} ({low_unknowns.iloc[0]['Unknown_Percentage']:.2f}%), {low_unknowns.iloc[1]['Station']} ({low_unknowns.iloc[1]['Unknown_Percentage']:.2f}%), {low_unknowns.iloc[2]['Station']} ({low_unknowns.iloc[2]['Unknown_Percentage']:.2f}%), and {low_unknowns.iloc[3]['Station']} ({low_unknowns.iloc[3]['Unknown_Percentage']:.2f}%) had the lowest unknown percentages, indicating effective detection.
"""
        table_data = df[['Station', 'Total Traffic', 'Unknown', 'Unknown_Percentage']].copy()
        table_data['Station'] = table_data['Station'].apply(self.normalize_station_name).str.title()
        table_data['Total Traffic'] = table_data['Total Traffic'].apply(lambda x: f"{int(x):,.0f}")
        table_data['Unknown'] = table_data['Unknown'].apply(lambda x: f"{int(x):,.0f}")
        table_data['Unknown_Percentage'] = table_data['Unknown_Percentage'].apply(lambda x: f"{x:.2f}%")
        
        return fig, table_data, analysis_text

    def generate_overloads_summary_by_station(self):
        """Generate overloads summary by station with enhanced bar charts and text"""
        if self.general_summary is None:
            self.wrangle_general_summary()
        
        # Create a copy of general_summary
        overloads_summary = self.general_summary.copy()
        overloads_summary['GVW OVERLOADS>0<2000'] = overloads_summary['Gross Vehicle Weight Overload>0<2000'].astype(int)
        overloads_summary['GVW OVERLOADS>=2000'] = overloads_summary['Gross Vehicle Weight Overload<2000'].astype(int)
        overloads_summary['TOTAL OVERLOADS'] = (
            overloads_summary['GVW OVERLOADS>0<2000'] +
            overloads_summary['GVW OVERLOADS>=2000']
        ).astype(int)
        overloads_summary['AVW OVERLOADS'] = overloads_summary['Axle Vehicle Weight Overload'].astype(int)
        overloads_summary['%COMPLIANCE'] = overloads_summary['Gross Vehicle Weight Compliance(%)'].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else "100.00%"
        )
        overloads_summary['TRAFFIC'] = overloads_summary['Total Traffic'].astype(int)
        
        # Prepare final table
        final_table = overloads_summary[[
            'Station', 'GVW OVERLOADS>0<2000', 'GVW OVERLOADS>=2000',
            'TOTAL OVERLOADS', 'AVW OVERLOADS', '%COMPLIANCE', 'TRAFFIC'
        ]]
        for col in ['GVW OVERLOADS>0<2000', 'GVW OVERLOADS>=2000',
                    'TOTAL OVERLOADS', 'AVW OVERLOADS', 'TRAFFIC']:
            final_table.loc[:, col] = final_table[col].apply(
                lambda x: f"{x:,.0f}" if pd.notnull(x) and x > 0 else "-"
            )
        
        final_table.insert(0, 'No', range(1, len(final_table) + 1))
        final_table = final_table.rename(columns={'Station': 'STATION'})
        final_table['STATION'] = final_table['STATION'].apply(self.normalize_station_name).str.title()
        
        def convert_to_float(value):
            if isinstance(value, str):
                value = value.replace(',', '').replace('-', '0')
            return float(value) if pd.notnull(value) else 0.0
        
        # Prepare plot table
        plot_table = final_table[final_table['STATION'] != 'Totals'].copy() if 'Totals' in final_table['STATION'].values else final_table.copy()
        plot_table = plot_table.reset_index(drop=True)
        plot_table['TOTAL_OVERLOADS_FLOAT'] = plot_table['TOTAL OVERLOADS'].apply(convert_to_float)
        plot_table['AVW_OVERLOADS_FLOAT'] = plot_table['AVW OVERLOADS'].apply(convert_to_float)
        plot_table['GVW_OVERLOADS_FLOAT'] = plot_table['GVW OVERLOADS>=2000'].apply(convert_to_float)
        
        for col in ['TOTAL_OVERLOADS_FLOAT', 'AVW_OVERLOADS_FLOAT', 'GVW_OVERLOADS_FLOAT']:
            plot_table[col] = pd.to_numeric(plot_table[col], errors='coerce').fillna(0.0)
        
        # Chart 1: Total GVW Overloads
        fig1 = plt.figure(figsize=(10, 8))
        plot_data = plot_table[plot_table['TOTAL_OVERLOADS_FLOAT'] > 0].sort_values('TOTAL_OVERLOADS_FLOAT', ascending=False)
        ax1 = sns.barplot(
            data=plot_data,
            y='STATION',
            x='TOTAL_OVERLOADS_FLOAT',
            palette='Blues_r',
            edgecolor='#333333',
            linewidth=0.8
        )
        plt.title('Gross Vehicle Weight Overloads by Station', fontsize=16, pad=20, weight='bold')
        plt.xlabel('Number of Overloads', fontsize=14)
        plt.ylabel('Station', fontsize=14)
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,.0f}"))
        ax1.tick_params(axis='y', labelsize=10)
        for p in ax1.patches:
            width = p.get_width()
            ax1.text(
                width + (plot_data['TOTAL_OVERLOADS_FLOAT'].max() * 0.02),
                p.get_y() + p.get_height()/2,
                f"{int(width):,.0f}",
                va='center',
                ha='left',
                fontsize=10,
                color='black',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3')
            )
        plt.tight_layout()
        
        top_overloads = plot_data.nlargest(3, 'TOTAL_OVERLOADS_FLOAT')
        low_overloads = plot_data[plot_data['TOTAL_OVERLOADS_FLOAT'] < 10].copy()
        compliance_rate = final_table['%COMPLIANCE'].str.rstrip('%').astype(float).mean()
        
        text1 = f"""
- {top_overloads.iloc[0]['STATION']} recorded the highest number of gross vehicle weight overloads, totaling {int(top_overloads.iloc[0]['TOTAL_OVERLOADS_FLOAT']):,.0f}.
- {top_overloads.iloc[1]['STATION']} and {top_overloads.iloc[2]['STATION']} also had substantial number of overloads, {int(top_overloads.iloc[1]['TOTAL_OVERLOADS_FLOAT']):,.0f} and {int(top_overloads.iloc[2]['TOTAL_OVERLOADS_FLOAT']):,.0f} vehicles respectively.
- Low overloads stations like {', '.join(low_overloads['STATION'].head(6).tolist()) if not low_overloads.empty else 'None'} recorded less than 10 cases of Gross vehicle weight overloads.
- On average, the compliance rate was {round(compliance_rate, 2):.2f}% compliance.
"""
        # Chart 2: Axle Weight Overloads
        fig2 = plt.figure(figsize=(10, 8))
        plot_data = plot_table[plot_table['AVW_OVERLOADS_FLOAT'] > 0].sort_values('AVW_OVERLOADS_FLOAT', ascending=False)
        ax2 = sns.barplot(
            data=plot_data,
            y='STATION',
            x='AVW_OVERLOADS_FLOAT',
            palette='Blues_r',
            edgecolor='#333333',
            linewidth=0.8
        )
        plt.title('Axle Vehicle Weight Overloads by Station', fontsize=16, pad=20, weight='bold')
        plt.xlabel('Number of Axle Overloads', fontsize=14)
        plt.ylabel('Station', fontsize=14)
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,.0f}"))
        ax2.tick_params(axis='y', labelsize=10)
        for p in ax2.patches:
            width = p.get_width()
            ax2.text(
                width + (plot_data['AVW_OVERLOADS_FLOAT'].max() * 0.02),
                p.get_y() + p.get_height()/2,
                f"{int(width):,.0f}",
                va='center',
                ha='left',
                fontsize=10,
                color='black',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3')
            )
        plt.tight_layout()
        
        total_axle_overloads = plot_table['AVW_OVERLOADS_FLOAT'].sum()
        top_axle = plot_data.iloc[0] if not plot_data.empty else pd.Series({'STATION': 'None', 'AVW_OVERLOADS_FLOAT': 0})
        low_axle = plot_data.nsmallest(3, 'AVW_OVERLOADS_FLOAT')
        
        text2 = f"""
This section focuses on vehicles with axle overloads.
- A total of {int(total_axle_overloads):,.0f} vehicles overloaded at the axle.
- {top_axle['STATION']} had the highest number of axle overloads, totaling {int(top_axle['AVW_OVERLOADS_FLOAT']):,.0f} vehicles.
- Low axle overload stations like {', '.join(low_axle['STATION'].tolist()) if not low_axle.empty else 'None'} recorded low numbers of axle overloads.
"""
        # Chart 3: Tagged Vehicles (GVW >2000kg)
        fig3 = plt.figure(figsize=(10, 8))
        plot_data = plot_table[plot_table['GVW_OVERLOADS_FLOAT'] > 0].sort_values('GVW_OVERLOADS_FLOAT', ascending=False)
        ax3 = sns.barplot(
            data=plot_data,
            y='STATION',
            x='GVW_OVERLOADS_FLOAT',
            palette='Blues_r',
            edgecolor='#333333',
            linewidth=0.8
        )
        plt.title('Tagged Vehicles (GVW >2000kg) by Station', fontsize=16, pad=20, weight='bold')
        plt.xlabel('Number of Tagged Vehicles', fontsize=14)
        plt.ylabel('Station', fontsize=14)
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,.0f}"))
        ax3.tick_params(axis='y', labelsize=10)
        for p in ax3.patches:
            width = p.get_width()
            ax3.text(
                width + (plot_data['GVW_OVERLOADS_FLOAT'].max() * 0.02),
                p.get_y() + p.get_height()/2,
                f"{int(width):,.0f}",
                va='center',
                ha='left',
                fontsize=10,
                color='black',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3')
            )
        plt.tight_layout()
        
        # Explicitly define df for total_traffic calculation
        df = self.general_summary[self.general_summary['Station'] != 'Totals'].copy() if 'Totals' in self.general_summary['Station'].values else self.general_summary.copy()
        if 'Total Traffic' not in df.columns:
            raise KeyError(f"'Total Traffic' column not found in general_summary. Available columns: {list(df.columns)}")
        
        total_tagged = plot_table['GVW_OVERLOADS_FLOAT'].sum()
        total_traffic = df['Total Traffic'].sum()
        tagged_percentage = round((total_tagged/total_traffic*100), 2) if total_traffic > 0 else 0.0
        top_tagged = plot_data.iloc[0] if not plot_data.empty else pd.Series({'STATION': 'None', 'GVW_OVERLOADS_FLOAT': 0})
        second_tagged = plot_data.iloc[1] if len(plot_data) > 1 else pd.Series({'STATION': 'None', 'GVW_OVERLOADS_FLOAT': 0})
        low_tagged = plot_data[plot_data['GVW_OVERLOADS_FLOAT'] < 5]
        zero_overloads = plot_table[plot_table['TOTAL_OVERLOADS_FLOAT'] == 0]['STATION'].tolist()
        
        text3 = f"""
This section examines vehicles with gross vehicle weight exceeding 2000 kg.
- A total of {int(total_tagged):,.0f} vehicles, accounting for {tagged_percentage:.2f}% of total traffic, were tagged for overloading.
- {top_tagged['STATION']} had the highest number of tags, totaling {int(top_tagged['GVW_OVERLOADS_FLOAT']):,.0f} vehicles.
- {second_tagged['STATION']} ({int(second_tagged['GVW_OVERLOADS_FLOAT']):,.0f} vehicles) also had notable tags.
- Additionally, {', '.join(low_tagged['STATION'].head(3).tolist()) if not low_tagged.empty else 'None'} recorded less than 5 tagged vehicles per station.
"""
        if zero_overloads:
            text3 += "- No overloaded vehicles were recorded at:\n"
            for i, station in enumerate(zero_overloads, 1):
                text3 += f"  - {station}\n"
        else:
            text3 += "- All stations recorded some overloads.\n"
        
        return [fig1, fig2, fig3], final_table, [text1, text2, text3]

    def generate_axle_config_summary(self):
        """Generate axle configuration summary with enhanced bar chart, table, and text"""
        if self.overloads_data is None:
            self.wrangle_overloads()
        
        df = self.overloads_data
        axle_counts = df['AXLE CONFIG'].value_counts().reset_index()
        axle_counts.columns = ['AXLE CONFIG', 'COUNT']
        total = int(axle_counts['COUNT'].sum())
        axle_counts['PERCENTAGE'] = [
            round((x/total*100), 2) if total > 0 else 0.0 for x in axle_counts['COUNT']
        ]
        axle_counts['PERCENTAGE'] = axle_counts['PERCENTAGE'].apply(lambda x: f"{x:.2f}%")
        axle_counts = axle_counts.sort_values('COUNT', ascending=False).reset_index(drop=True)
        
        total_row = pd.DataFrame({
            'AXLE CONFIG': [''],
            'COUNT': [total],
            'PERCENTAGE': ['100.00%']
        })
        axle_counts = pd.concat([axle_counts, total_row], ignore_index=True)
        
        plot_data = axle_counts[axle_counts['AXLE CONFIG'] != ''].copy()
        plot_data['COUNT'] = plot_data['COUNT'].astype(int)
        plot_data['PERCENTAGE_VALUE'] = plot_data['PERCENTAGE'].str.rstrip('%').astype(float)
        
        # Create enhanced bar chart
        fig = plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=plot_data,
            x='COUNT',
            y='AXLE CONFIG',
            palette='Blues_r',
            edgecolor='#333333',
            linewidth=0.8
        )
        plt.title('GVW Overloads >2000kg by Axle Configuration', fontsize=16, pad=20, weight='bold')
        plt.xlabel('Number of Overloads', fontsize=14)
        plt.ylabel('Axle Configuration', fontsize=14)
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,.0f}"))
        ax.tick_params(axis='y', labelsize=10)
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            percentage = plot_data.iloc[i]['PERCENTAGE_VALUE']
            ax.text(
                width + (plot_data['COUNT'].max() * 0.02),
                p.get_y() + p.get_height()/2,
                f"{int(width):,.0f} ({percentage:.1f}%)",
                va='center',
                ha='left',
                fontsize=10,
                color='black',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3')
            )
        plt.tight_layout()
        
        total_overloads = axle_counts[axle_counts['AXLE CONFIG'] == '']['COUNT'].values[0]
        top_config = plot_data.iloc[0]
        second_config = plot_data.iloc[1] if len(plot_data) > 1 else pd.Series({'AXLE CONFIG': 'None', 'COUNT': 0, 'PERCENTAGE_VALUE': 0.0})
        third_config = plot_data.iloc[2] if len(plot_data) > 2 else pd.Series({'AXLE CONFIG': 'None', 'COUNT': 0, 'PERCENTAGE_VALUE': 0.0})
        other_configs = plot_data[plot_data['PERCENTAGE_VALUE'] < 2]
        moderate_config = plot_data[(plot_data['PERCENTAGE_VALUE'] >= 2) &
                            (plot_data['PERCENTAGE_VALUE'] <= 10)].iloc[0] if not plot_data[(plot_data['PERCENTAGE_VALUE'] >= 2) &
                            (plot_data['PERCENTAGE_VALUE'] <= 10)].empty else pd.Series({'AXLE CONFIG': 'None', 'COUNT': 0, 'PERCENTAGE_VALUE': 0.0})
        
        analysis_text = f"""
Table 8 contains the distribution of overloads per Axle configuration. Overall:
* Most tagged vehicles were {top_config['AXLE CONFIG']} trucks ({top_config['COUNT']:,.0f}, {top_config['PERCENTAGE_VALUE']:.2f}%).
* {second_config['AXLE CONFIG']} and {third_config['AXLE CONFIG']} also had notable overloads, {second_config['COUNT']:,.0f} and {third_config['COUNT']:,.0f}, respectively.
* Other vehicle category which had moderately high overloads was {moderate_config['AXLE CONFIG']} ({moderate_config['COUNT']:,.0f}).
* Other axle configurations ({', '.join(other_configs['AXLE CONFIG'].tolist()) if not other_configs.empty else 'None'}) had percentage of GVW overloads below 2%.
"""
        return fig, axle_counts, analysis_text

    def generate_unknown_tags_analysis(self):
        """Generate unknown tags analysis with enhanced stacked bar chart, table, sign-off table, and text"""
        if self.overloads_data is None:
            self.wrangle_overloads()
        
        df = self.overloads_data.copy()
        station_col = 'STATION___' if 'STATION___' in df.columns else next((col for col in df.columns if col.upper().startswith('STATION')), None)
        if station_col is None:
            raise KeyError(f"No station column found in Sheet2. Available columns: {list(df.columns)}")
        
        df = df.rename(columns={station_col: 'STATION'})
        df['Station'] = df['STATION'].apply(self.normalize_station_name)
        df['Station'] = df['Station'].str.split('-').str[0].str.title()
        
        unknown_tags = df[df['REGISTRATION'].str.contains('UNKNOWN', case=False, na=False)].groupby('Station').size().reset_index(name='UNKNOWN TAGS')
        total_tags = df.groupby('Station').size().reset_index(name='TOTAL TAGS')
        result = pd.merge(unknown_tags, total_tags, on='Station', how='right').fillna({'UNKNOWN TAGS': 0})
        result['UNKNOWN TAGS'] = result['UNKNOWN TAGS'].astype(int)
        result['TOTAL TAGS'] = result['TOTAL TAGS'].astype(int)
        result['KNOWN TAGS'] = (result['TOTAL TAGS'] - result['UNKNOWN TAGS']).astype(int)
        result['% UNKNOWN'] = result.apply(lambda x: round((x['UNKNOWN TAGS']/x['TOTAL TAGS']*100), 1) if x['TOTAL TAGS'] > 0 else 0.0, axis=1)
        
        result = result.sort_values('% UNKNOWN', ascending=False).reset_index(drop=True)
        
        grand_total = pd.DataFrame({
            'Station': ['GRAND TOTAL'],
            'KNOWN TAGS': [result['KNOWN TAGS'].sum()],
            'UNKNOWN TAGS': [result['UNKNOWN TAGS'].sum()],
            'TOTAL TAGS': [result['TOTAL TAGS'].sum()],
            '% UNKNOWN': [round((result['UNKNOWN TAGS'].sum()/result['TOTAL TAGS'].sum()*100), 1) if result['TOTAL TAGS'].sum() > 0 else 0.0]
        })
        
        final_table = pd.concat([result, grand_total], ignore_index=True)
        
        final_table['KNOWN TAGS'] = final_table['KNOWN TAGS'].apply(lambda x: f"{int(x):,.0f}")
        final_table['UNKNOWN TAGS'] = final_table['UNKNOWN TAGS'].apply(lambda x: f"{int(x):,.0f}")
        final_table['TOTAL TAGS'] = final_table['TOTAL TAGS'].apply(lambda x: f"{int(x):,.0f}")
        final_table['% UNKNOWN'] = final_table['% UNKNOWN'].apply(lambda x: f"{x:.1f}%")
        final_table = final_table[['Station', 'KNOWN TAGS', 'UNKNOWN TAGS', 'TOTAL TAGS', '% UNKNOWN']]
        
        # Create enhanced stacked bar chart with percentages
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_data = final_table[final_table['Station'] != 'GRAND TOTAL'].copy()
        
        plot_data['UNKNOWN TAGS_NUM'] = plot_data['UNKNOWN TAGS'].str.replace(',', '').astype(int)
        plot_data['KNOWN TAGS_NUM'] = plot_data['KNOWN TAGS'].str.replace(',', '').astype(int)
        plot_data['TOTAL TAGS_NUM'] = plot_data['TOTAL TAGS'].str.replace(',', '').astype(int)
        plot_data['KNOWN_PCT'] = (plot_data['KNOWN TAGS_NUM'] / plot_data['TOTAL TAGS_NUM'] * 100).round(1)
        plot_data['UNKNOWN_PCT'] = (plot_data['UNKNOWN TAGS_NUM'] / plot_data['TOTAL TAGS_NUM'] * 100).round(1)
        
        ax.barh(
            plot_data['Station'],
            plot_data['KNOWN_PCT'],
            color='#2ca02c',
            label='Known Registration',
            edgecolor='#333333',
            linewidth=0.8
        )
        ax.barh(
            plot_data['Station'],
            plot_data['UNKNOWN_PCT'],
            left=plot_data['KNOWN_PCT'],
            color='#d62728',
            label='Unknown Registration',
            edgecolor='#333333',
            linewidth=0.8
        )
        
        for i, (station, known_pct, unknown_pct, unknown_count, total) in enumerate(zip(
                plot_data['Station'],
                plot_data['KNOWN_PCT'],
                plot_data['UNKNOWN_PCT'],
                plot_data['UNKNOWN TAGS_NUM'],
                plot_data['TOTAL TAGS_NUM']
            )):
            ax.text(
                known_pct + unknown_pct + 2,
                i,
                f"{unknown_count:,.0f} ({unknown_pct:.1f}%)",
                va='center',
                ha='left',
                fontsize=10,
                color='black',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3')
            )
        
        ax.set_title('Vehicle Registration Status by Station', fontsize=16, pad=20, weight='bold')
        ax.set_xlabel('Percentage of Vehicles (%)', fontsize=14)
        ax.set_ylabel('Station', fontsize=14)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.set_xlim(0, 140)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
        ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor='#333333')
        
        summary_text = (f"Total Unknown: {grand_total['UNKNOWN TAGS'].iloc[0]:,.0f} "
                        f"({grand_total['% UNKNOWN'].iloc[0]:.1f}%) \n"
                        f"Total Vehicles: {grand_total['TOTAL TAGS'].iloc[0]:,.0f}")
        ax.annotate(
            summary_text,
            xy=(0.98, 0.02),
            xycoords='axes fraction',
            ha='right',
            va='bottom',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#333333', boxstyle='round,pad=0.3')
        )
        
        plt.tight_layout()
        
        sign_off_table = pd.DataFrame({
            'Role': ['PREPARED BY:', 'CHECKED BY:', 'APPROVED BY:'],
            'Name': ['[Name]', '[Name]', '[Name]']
        })
        
        total_unknown = int(final_table['UNKNOWN TAGS'].iloc[-1].replace(',', ''))
        total_tags = int(final_table['TOTAL TAGS'].iloc[-1].replace(',', ''))
        top_unknown = final_table.iloc[0]
        high_unknown = final_table[final_table['% UNKNOWN'].str.rstrip('%').astype(float) > 50]
        moderate_unknown = final_table[(final_table['% UNKNOWN'].str.rstrip('%').astype(float) >= 10) &
                                      (final_table['% UNKNOWN'].str.rstrip('%').astype(float) <= 50)]
        
        # Use UNKNOWN TAGS_NUM from plot_data for top_unknown's station
        top_unknown_count = plot_data[plot_data['Station'] == top_unknown['Station']]['UNKNOWN TAGS_NUM'].iloc[0]
        
        analysis_text = f"""
A total of {total_unknown:,.0f} tagged vehicles ({grand_total['% UNKNOWN'].iloc[0]:.1f}%) had unidentified plates across all stations.
Key Findings:
* {top_unknown['Station']} had the highest absolute number of unknown plates ({top_unknown['UNKNOWN TAGS']}), representing {top_unknown['% UNKNOWN']} of its total tags.
* Stations with >50% unknown rate: {', '.join(high_unknown['Station'].tolist()) if not high_unknown.empty else 'None'}
* Stations with moderate (10-50%) unknown rate: {', '.join(moderate_unknown['Station'].tolist()) if not moderate_unknown.empty else 'None'}

Operational Insights:
* {top_unknown['Station']} station alone accounts for {round((top_unknown_count/total_unknown*100), 2):.2f}% of all unknown tags.
* {len(high_unknown)} stations show critical identification issues (>50% unknown rate).
* {len(plot_data[plot_data['UNKNOWN TAGS_NUM'] == 0])} stations had perfect identification (0 unknown tags).
"""
        return fig, final_table, sign_off_table, analysis_text