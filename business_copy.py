import sys
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
        # CORRECT INDIVIDUAL STATION COMPLIANCE CALCULATION
        df['Total_overloads'] = df['Gross Vehicle Weight Overload>0<2000'] + df['Gross Vehicle Weight Overload<2000']
        df['Gross Vehicle Weight Compliance(%)'] = np.where(
            df['Total Traffic'] > 0,
            ((df['Total Traffic'] - df['Total_overloads']) / df['Total Traffic']) * 100,
            100.0
        ).round(2)
        numerical_cols = [col for col in df.columns if col not in ['Station', 'Date', 'Gross Vehicle Weight Compliance(%)']]
        totals = df[numerical_cols].sum()
        # CORRECT OVERALL COMPLIANCE CALCULATION FOR TOTALS ROW
        total_compliance = ((totals['Total Traffic'] - totals['Total_overloads']) / totals['Total Traffic'] * 100).round(2) if totals['Total Traffic'] > 0 else 100.0
        
        totals_row = pd.DataFrame([{
            'Station': 'Totals',
            'Date': '',
            **{col: totals[col] for col in numerical_cols},
            'Gross Vehicle Weight Compliance(%)': total_compliance
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
    def generate_general_summary_table(self):
        """Generate General Summary DataFrame with multi-level columns for display only"""
        # Define multi-level column structure
        columns = pd.MultiIndex.from_tuples([
            ('Station', ''),
            ('Date', ''),
            ('Unknown', ''),
            ('Passenger Car', ''),
            ('Trucks Only', '2 Axles'),
            ('Trucks Only', '3 Axles'),
            ('Trucks Only', '4 Axles'),
            ('Trucks Only', '5 Axles'),
            ('Trucks Only', '6 Axles'),
            ('Trucks Only', '7 Axles'),
            ('Trucks Only', '8 Axles'),
            ('Buses Only', '2 Axles'),
            ('Buses Only', '3 Axles'),
            ('Buses Only', 'Total Buses'),
            ('Overload', 'Axle Vehicle Weight'),
            ('Overload', 'Gross Vehicle Weight >0<2000'),
            ('Overload', 'Gross Vehicle Weight >=2000'),
            ('Overload', 'No Gross Vehicle Weight'),
            ('Compliance', 'Gross Vehicle Weight (%)'),
            ('Traffic', 'Total')
        ])
        
        # Read the data
        df = pd.read_excel(self.file_path, sheet_name=self.sheet1_name, header=None, usecols=range(1, len(columns)+1))
        if len(df.columns) != len(columns):
            raise ValueError(f"Expected {len(columns)} columns but found {len(df.columns)}")
        
        df.columns = columns
        
        # Remove Date column as requested
        df = df.drop(columns=[('Date', '')])
        
        # Convert station names to proper case
        df[('Station', '')] = df[('Station', '')].apply(
            lambda x: x.title() if isinstance(x, str) else x
        )
        
        # Calculate compliance for each station
        df[('Overload', 'Total Overloads')] = (df[('Overload', 'Gross Vehicle Weight >0<2000')] + 
                                            df[('Overload', 'Gross Vehicle Weight >=2000')])
        
        df[('Compliance', 'Gross Vehicle Weight (%)')] = np.where(
            df[('Traffic', 'Total')] > 0,
            ((df[('Traffic', 'Total')] - df[('Overload', 'Total Overloads')]) / 
            df[('Traffic', 'Total')]) * 100,
            100.0
        ).round(2)
        
        # Get numerical columns for summing (excluding Station and compliance percentage)
        numerical_cols = [col for col in df.columns 
                        if col not in [('Station', ''), ('Compliance', 'Gross Vehicle Weight (%)')]]
        
        totals = df[numerical_cols].sum()
        
        # Calculate overall compliance for totals row
        total_traffic = totals[('Traffic', 'Total')]
        total_overloads = totals[('Overload', 'Total Overloads')]
        total_compliance = ((total_traffic - total_overloads) / total_traffic * 100).round(2) if total_traffic > 0 else 100.0
        
        # Create totals row
        totals_dict = {('Station', ''): 'Totals'}
        for col in numerical_cols:
            totals_dict[col] = totals[col]
        totals_dict[('Compliance', 'Gross Vehicle Weight (%)')] = total_compliance
        
        totals_row = pd.DataFrame([totals_dict])
        totals_row.columns = pd.MultiIndex.from_tuples(totals_row.columns)
        
        # Remove the temporary total overloads column
        df = df.drop(columns=[('Overload', 'Total Overloads')])
        
        # Concatenate with proper column alignment
        df = pd.concat([df, totals_row], ignore_index=True)
        
        # Format numerical values with thousand separators and no decimals
        numerical_columns_to_format = [
            ('Unknown', ''),
            ('Passenger Car', ''),
            ('Trucks Only', '2 Axles'),
            ('Trucks Only', '3 Axles'),
            ('Trucks Only', '4 Axles'),
            ('Trucks Only', '5 Axles'),
            ('Trucks Only', '6 Axles'),
            ('Trucks Only', '7 Axles'),
            ('Trucks Only', '8 Axles'),
            ('Buses Only', '2 Axles'),
            ('Buses Only', '3 Axles'),
            ('Buses Only', 'Total Buses'),
            ('Overload', 'Axle Vehicle Weight'),
            ('Overload', 'Gross Vehicle Weight >0<2000'),
            ('Overload', 'Gross Vehicle Weight >=2000'),
            ('Overload', 'No Gross Vehicle Weight'),
            ('Traffic', 'Total')
        ]
        
        for col in numerical_columns_to_format:
            if col in df.columns:
                # Replace NaN/None with empty string
                df[col] = df[col].fillna('')
                # Format numbers with thousand separators and no decimals
                df[col] = df[col].apply(
                    lambda x: f"{int(x):,}" if isinstance(x, (int, float)) and not pd.isna(x) and x != '' else x
                )
        
        # Format compliance percentage to one decimal place
        compliance_col = ('Compliance', 'Gross Vehicle Weight (%)')
        if compliance_col in df.columns:
            df[compliance_col] = df[compliance_col].apply(
                lambda x: f"{float(x):.1f}%" if isinstance(x, (int, float)) and not pd.isna(x) and x != '' else x
            )
        
        # Replace any remaining nulls/NaNs with blank strings
        df = df.fillna('')
        df = df.drop(columns=[('Overload', 'Total Overloads')])
        return df
    
    def generate_traffic_summary(self):
        """Generate traffic summary DataFrame and text"""
        if self.general_summary is None:
            self.wrangle_general_summary()
        
        df = self.general_summary.iloc[:-1] if 'Totals' in self.general_summary['Station'].values else self.general_summary
        
        # Calculate counts as integers first
        counts = [
            int(df['Unknown'].sum()),
            int(df['Passengers car'].sum()),
            int(df[['2-axle trucks', '3-axle trucks', '4-axle trucks', '5-axle trucks',
                '6-axle trucks', '7-axle trucks', '8-axle trucks']].sum().sum()),
            int(df[['2-axle buses', '3-axle buses']].sum().sum()),
            int(df['Total Traffic'].sum())
        ]
        
        traffic_summary = pd.DataFrame({
            'Vehicles Category': ['Unknown', 'Passenger Cars', 'Trucks', 'Buses', 'Total Traffic'],
            'Count': ["{:,}".format(count) for count in counts]  # Format with commas here
        })
        
        # Use original integer values for calculations
        total_traffic = counts[4]
        relative_percentages = [
            round((x/total_traffic*100), 2) if total_traffic > 0 else 0.0 for x in counts
        ]
        traffic_summary['Percentage(%)'] = [
            f"{x:.2f}%" for x in relative_percentages
        ]
        # For the text summary, use the original integer values
        total_vehicles = counts[4]
        passenger_cars = counts[1]
        trucks = counts[2]
        unknown = counts[0]
        buses = counts[3]
        passenger_pct = round((passenger_cars/total_vehicles*100), 2) if total_vehicles > 0 else 0.0
        trucks_pct = round((trucks/total_vehicles*100), 2) if total_vehicles > 0 else 0.0
        unknown_pct = round((unknown/total_vehicles*100), 2) if total_vehicles > 0 else 0.0
        buses_pct = round((buses/total_vehicles*100), 2) if total_vehicles > 0 else 0.0
        summary_text = f"""
  Total number of vehicles weighed across the stations: {total_vehicles:,.0f}
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
        
        # Calculate raw counts first (keep as integers)
        raw_counts = [
            int(df['Gross Vehicle Weight Overload>0<2000'].sum()),
            int(df['Gross Vehicle Weight Overload<2000'].sum()),
            int(df['No gross vehicle Weight Overloads'].sum()),
            int(df['Total Traffic'].sum())
        ]
        
        # Create DataFrame with formatted display values
        overloads_summary = pd.DataFrame({
            'Category': [
                'Gross Vehicle Weight Overload >0 <2000',
                'Gross Vehicle Weight Overload >=2000',
                'No Gross Vehicle Weight Overload',
                'TOTAL TRAFFIC'
            ],
            'Number of Vehicles': ["{:,}".format(count) for count in raw_counts],
            'Percentage (%)': [
                f"{(raw_counts[0]/raw_counts[3]*100):.2f}%" if raw_counts[3] > 0 else "0.00%",
                f"{(raw_counts[1]/raw_counts[3]*100):.2f}%" if raw_counts[3] > 0 else "0.00%",
                f"{(raw_counts[2]/raw_counts[3]*100):.2f}%" if raw_counts[3] > 0 else "0.00%",
                "100.00%"
            ]
        })
        
        # Generate text summary using raw counts (not the formatted strings)
        overload_text = f"""
    - Gross vehicle weight overload between 0 and 2,000 kg: {raw_counts[0]:,} ({overloads_summary.loc[0, 'Percentage (%)']})
    - Gross vehicle weight exceeding 2,000 kg: {raw_counts[1]:,} ({overloads_summary.loc[1, 'Percentage (%)']})
    - Compliance - Number of vehicles with no GVW overload: {raw_counts[2]:,} ({overloads_summary.loc[2, 'Percentage (%)']})
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
        top3_percentage = round((top_stations['Total Traffic'].sum()/total_traffic*100), 2)
        if top3_percentage > 50:
            concentration_level = "highly concentrated"
        elif top3_percentage > 30:
            concentration_level = "moderately concentrated"
        else:
            concentration_level = "well distributed"
        
        analysis_text = f"""
            Traffic distribution analysis reveals {concentration_level} patterns across monitoring stations. {top_stations.iloc[0]['Station']} recorded the highest volume with {int(top_stations.iloc[0]['Total Traffic']):,.0f} vehicles ({round((top_stations.iloc[0]['Total Traffic']/total_traffic*100), 2):.2f}% of total traffic), followed by {top_stations.iloc[1]['Station']} ({int(top_stations.iloc[1]['Total Traffic']):,.0f}) and {top_stations.iloc[2]['Station']} ({int(top_stations.iloc[2]['Total Traffic']):,.0f}).

            The top three stations collectively accounted for {top3_percentage:.2f}% of all traffic, while remaining stations showed volumes ranging from {int(mid_range_stats['min']):,.0f} to {int(mid_range_stats['max']):,.0f}. {bottom_station.iloc[0]['Station']} recorded the lowest traffic at {int(bottom_station.iloc[0]['Total Traffic']):,.0f} vehicles, representing {round((bottom_station.iloc[0]['Total Traffic']/total_traffic*100), 2):.2f}% of total network traffic.
            """
        return fig, station_summary, analysis_text

    def generate_trucks_distribution(self):
        """Generate trucks distribution analysis with enhanced donut chart, table, and text"""
        if self.general_summary is None:
            self.wrangle_general_summary()
        
        df = self.general_summary.iloc[:-1] if 'Totals' in self.general_summary['Station'].values else self.general_summary.copy()
        
        truck_categories = {
            '2 Axle': '2-axle trucks', '3 Axle': '3-axle trucks', '4 Axle': '4-axle trucks',
            '5 Axle': '5-axle trucks','7 Axle': '7-axle trucks', '6 Axle': '6-axle trucks',
            '8 Axle': '8-axle trucks'
        }
        counts = [int(df[col].sum()) for col in truck_categories.values()]
        total_trucks = sum(counts)
        percentages = [round((x/total_trucks*100), 2) if total_trucks > 0 else 0.0 for x in counts]
        
        average_axles = sum([(i+2)*count for i, count in enumerate(counts)])/total_trucks if total_trucks > 0 else 0.0
        heavy_truck_pct = sum(percentages[3:7])
        top_config_pct = percentages[0] + percentages[5]
        
        analysis_text = f"""
Truck Fleet Composition Analysis:
• 2-Axle Configuration: {percentages[0]:.1f}% ({counts[0]:,.0f} vehicles) - Predominant vehicle type
• 6-Axle Configuration: {percentages[5]:.1f}% ({counts[5]:,.0f} vehicles) - Primary heavy freight carriers
• Supplementary Configurations: 
  • 3-Axle: {percentages[1]:.1f}%
  • 4-Axle: {percentages[2]:.1f}%
  • 5-Axle: {percentages[3]:.1f}%
  • 7-Axle: {percentages[4]:.1f}%
  • 8-Axle: {'Not operational' if counts[6] == 0 else f'{percentages[6]:.1f}%'}
• Dominant Pair: 2-axle and 6-axle configurations represent {top_config_pct:.1f}% of total truck volume
• Heavy-Duty Segment: Vehicles with 5+ axles constitute {heavy_truck_pct:.1f}% of truck fleet
"""
        # Create enhanced donut chart
        fig = plt.figure(figsize=(6, 4))
        palette = sns.color_palette("Set2", len(truck_categories))
        wedges, texts, autotexts = plt.pie(
            counts,
            labels=[f"{label} ({count:,.0f})" for label, count in zip(truck_categories.keys(), counts)],
            autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
            startangle=90,
            colors=palette,
            wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1),
            textprops={'fontsize': 10},
            pctdistance=0.85
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        plt.gca().add_artist(centre_circle)
        plt.title('Truck Traffic Distribution by Axle Configuration', fontsize=14, pad=18, weight='bold')
        plt.axis('equal')
        plt.tight_layout()
        
        trucks_distribution = pd.DataFrame({
            'Axle Configuration': list(truck_categories.keys()) + ['TOTAL TRUCKS'],
            'Total Count': [f"{x:,.0f}" for x in counts] + [f"{total_trucks:,.0f}"],
            'Percentage': [f"{x:.2f}%" for x in percentages] + ["100.00%"]
        })
        
        return fig, trucks_distribution, analysis_text

    def generate_vehicle_category_analysis(self):
        """Generate vehicle category analysis with bar chart, table, and text"""
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
        
        # Analysis text (unchanged)
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
Traffic Composition Analysis:
• Primary Vehicle Category: {top_category} represents {top_percentage:.1f}% of total traffic volume
• Truck Segment Overview: All trucks constitute {truck_percentage:.1f}% of traffic, predominantly comprised of:
  - 2-axle trucks: {percentages['2-axle trucks']:.1f}%
  - 6-axle trucks: {percentages['6-axle trucks']:.1f}%
• Passenger Vehicles: Cars account for {percentages['Passenger cars']:.1f}% of total traffic
• Public Transport: Buses represent {bus_percentage:.1f}% of vehicle volume
• Unclassified Vehicles: Unknown vehicle types constitute {unknown_percentage:.1f}%
• Category Concentration: Top three vehicle categories collectively represent {sum(sorted(percentages.values(), reverse=True)[:3]):.1f}% of total traffic
"""

        fig10 = plt.figure(figsize=(12, 8))  # Optimized for Word docs
        ax = plt.gca()
        
        # Sort data - highest percentage first (top position)
        viz_data = pd.DataFrame({
            'Category': list(categories.keys()),
            'Count': list(categories.values()),
            'Percentage': list(percentages.values())
        }).sort_values('Percentage', ascending=True)  # Critical: ascending=True for vertical order
        
        # Create horizontal bars
        palette = sns.color_palette("Set2", len(categories))
        bars = plt.barh(
            y=viz_data['Category'],
            width=viz_data['Percentage'],
            color=palette,
            edgecolor='white',
            linewidth=1,
            height=0.7
        )
        
        # Professional chart styling
        plt.title('Vehicle Category Distribution', fontsize=22, pad=18, weight='bold')
        plt.xlabel('Percentage of Total Traffic (%)', fontsize=16)
        plt.ylabel('')
        plt.yticks(fontsize=16)
        plt.xlim(0, max(viz_data['Percentage']) * 1.2)
        
        # Value labels
        for bar, count, pct in zip(bars, viz_data['Count'], viz_data['Percentage']):
            plt.text(
                bar.get_width() + 0.8,
                bar.get_y() + bar.get_height()/2,
                f"{count:,.0f} ({pct:.1f}%)",
                va='center',
                ha='left',
                fontsize=16,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.5)
            )
        
        # Clean axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.grid(axis='x', linestyle=':', alpha=0.4)
        plt.tight_layout()
        
        # --- TABLE --- (Separate dataframe)
        vehicle_distribution = pd.DataFrame({
            'Vehicle Category': list(categories.keys()),
            'Total Count': [f"{v:,.0f}" for v in categories.values()],
            'Percentage': [f"{v:.2f}%" for v in percentages.values()]
        }).sort_values('Percentage', ascending=False)  # Descending for table
        
        # Return objects for Word document insertion
        return fig10, vehicle_distribution, analysis_text

    def generate_unknowns_analysis(self):
        """Generate unknowns analysis with professional formatting and optimal readability"""
        if self.general_summary is None:
            self.wrangle_general_summary()
        
        # Prepare data with properly formatted station names
        df = self.general_summary[self.general_summary['Station'] != 'Totals'].copy() if 'Totals' in self.general_summary['Station'].values else self.general_summary.copy()
        
        # Format station names to Title Case (e.g., "Eldama Ravine" instead of "ELDAMA RAVINE")
        df['Station'] = df['Station'].str.title()
        
        # Calculate unknown percentage
        df['Unknown_Percentage'] = np.where(
            df['Total Traffic'] > 0,
            ((df['Unknown'] / df['Total Traffic']) * 100).round(2),
            0.0
        ).astype(float)
        
        # Sort by unknown percentage then traffic volume
        df = df.sort_values(['Unknown_Percentage', 'Total Traffic'], ascending=[False, False])
        
        # Get top and bottom performers
        top_unknowns = df.nlargest(3, 'Unknown_Percentage')
        low_unknowns = df.nsmallest(4, 'Unknown_Percentage')
        
        # Configure plot settings for optimal readability
        plt.style.use('seaborn-v0_8')
        plt.rcParams.update({
            'figure.dpi': 120,
            'savefig.dpi': 300,
            'font.size': 14,
            'axes.titlesize': 24,
            'axes.labelsize': 20,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16,
            'font.family': 'Arial'
        })

        # Create figure with optimal dimensions
        fig, ax1 = plt.subplots(figsize=(16, 10))
        
        # Bar chart for Total Traffic with proper station names
        station_names = df['Station']
        bar_colors = sns.color_palette('magma', n_colors=len(df))
        bars = ax1.bar(
            station_names,
            df['Total Traffic'],
            color=bar_colors,
            edgecolor='#1a1a1a',
            linewidth=0.6,
            alpha=0.8,
            width=0.7
        )

        # Secondary y-axis for Unknown_Percentage
        ax2 = ax1.twinx()
        line = ax2.plot(
            station_names,
            df['Unknown_Percentage'],
            color='#ff4d4d',
            linestyle='-',
            linewidth=3,
            marker='o',
            markersize=8,
            markeredgecolor='#1a1a1a',
            markeredgewidth=0.8,
            label='Unknown Vehicles (%)'
        )
        # Add 20% quality threshold line
        ax2.axhline(
            y=20, 
            color='#ff4d4d', 
            linestyle='--', 
            linewidth=2.5, 
            label='20% Quality Threshold'
        )

        # Customize axes with proper labels
        ax1.set_xlabel('Station', fontsize=20, fontweight='bold', labelpad=15)
        ax1.set_ylabel('Total Traffic (Vehicles)', fontsize=20, fontweight='bold', labelpad=15)
        ax2.set_ylabel('Unknown Vehicles (%)', fontsize=20, fontweight='bold', color='#ff4d4d', labelpad=15)
        
        # Title with improved spacing
        ax1.set_title(
            'Traffic Volume and Unknown Vehicle Percentage by Station',
            fontsize=24,
            pad=20,
            fontweight='bold'
        )

        # Format axes
        ax1.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.35, axis='y')
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 10000 else f"{int(x):,.0f}"))
        
        # Set y-axis limits with buffer
        ax1.set_ylim(0, df['Total Traffic'].max() * 1.15)
        ax2.set_ylim(0, min(120, df['Unknown_Percentage'].max() * 1.25))
        
        # Rotated x-tick labels with proper alignment and formatting
        ax1.set_xticks(range(len(station_names)))
        ax1.set_xticklabels(
            station_names,
            rotation=45,
            ha='right',
            rotation_mode='anchor',
            fontsize=14
        )

        # Add combined legend
        lines, labels = ax2.get_legend_handles_labels()
        ax2.legend(
            lines, 
            labels, 
            loc='upper right', 
            fontsize=16,
            frameon=True,
            framealpha=0.9,
            edgecolor='#1a1a1a',
            bbox_to_anchor=(1.0, 1.0)
        )
        # Adjust layout with more padding
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(
            bottom=0.25,  # Extra space for x-labels
            top=0.9,      # Space for title
            left=0.12,    # Space for y-label
            right=0.88    # Space for secondary y-axis
        )
            # Generate analysis text with properly formatted station names
        analysis_text = f"""
Unknown Vehicle Distribution Analysis:

• This assessment examines the distribution of unclassified vehicles across 
  monitoring stations, demonstrating significant variability influenced by 
  geographical location, traffic volume patterns, and detection system 
  capabilities.

• Stations with High Unknown Percentages (>20%):
  • {top_unknowns.iloc[0]['Station']} ({top_unknowns.iloc[0]['Unknown_Percentage']:.1f}%)
  • {top_unknowns.iloc[1]['Station']} ({top_unknowns.iloc[1]['Unknown_Percentage']:.1f}%)
  • {top_unknowns.iloc[2]['Station']} ({top_unknowns.iloc[2]['Unknown_Percentage']:.1f}%)

• Stations with Excellent Detection Performance (<10% Unknowns):
  • {low_unknowns.iloc[0]['Station']} ({low_unknowns.iloc[0]['Unknown_Percentage']:.1f}%)
  • {low_unknowns.iloc[1]['Station']} ({low_unknowns.iloc[1]['Unknown_Percentage']:.1f}%)
  • {low_unknowns.iloc[2]['Station']} ({low_unknowns.iloc[2]['Unknown_Percentage']:.1f}%)
  • {low_unknowns.iloc[3]['Station']} ({low_unknowns.iloc[3]['Unknown_Percentage']:.1f}%)
"""
        
        # Prepare table data with consistent formatting
        table_data = df[['Station', 'Total Traffic', 'Unknown', 'Unknown_Percentage']].copy()
        table_data['Total Traffic'] = table_data['Total Traffic'].apply(lambda x: f"{int(x):,.0f}")
        table_data['Unknown'] = table_data['Unknown'].apply(lambda x: f"{int(x):,.0f}")
        table_data['Unknown_Percentage'] = table_data['Unknown_Percentage'].apply(lambda x: f"{x:.1f}%")
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
        
        # CORRECT COMPLIANCE RATE CALCULATION
        # Use the original df (general_summary without Totals row) for accurate totals
        df = self.general_summary[self.general_summary['Station'] != 'Totals'].copy() if 'Totals' in self.general_summary['Station'].values else self.general_summary.copy()
        
        total_overloads = df['Gross Vehicle Weight Overload>0<2000'].sum() + df['Gross Vehicle Weight Overload<2000'].sum()
        total_traffic = df['Total Traffic'].sum()
        compliance_rate = (1 - (total_overloads / total_traffic)) * 100 if total_traffic > 0 else 100.0
        
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
        
        text1 = f"""
• {top_overloads.iloc[0]['STATION']} recorded the highest number of gross vehicle weight 
  overloads, totaling {int(top_overloads.iloc[0]['TOTAL_OVERLOADS_FLOAT']):,.0f} incidents.

• {top_overloads.iloc[1]['STATION']} and {top_overloads.iloc[2]['STATION']} also reported 
  substantial overload volumes with {int(top_overloads.iloc[1]['TOTAL_OVERLOADS_FLOAT']):,.0f} 
  and {int(top_overloads.iloc[2]['TOTAL_OVERLOADS_FLOAT']):,.0f} vehicles respectively.

• Multiple stations including {', '.join(low_overloads['STATION'].head(6).tolist()) 
  if not low_overloads.empty else 'various locations'} demonstrated excellent compliance, 
  recording fewer than 10 cases of gross vehicle weight overloads.

• Overall network compliance reached {compliance_rate:.2f}%, representing 
  {total_traffic - total_overloads:,.0f} compliant vehicles out of {total_traffic:,.0f} 
  total vehicles monitored.
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
Axle Overload Analysis: This section examines vehicles exceeding individual axle weight limits, 
with {int(total_axle_overloads):,.0f} total axle overload incidents recorded across the network. 
{top_axle['STATION']} reported the highest incidence with {int(top_axle['AVW_OVERLOADS_FLOAT']):,.0f} 
cases of axle weight violations.

Several stations demonstrated excellent compliance with minimal axle overload activity, 
including {', '.join(low_axle['STATION'].tolist()) if not low_axle.empty else 'multiple locations'}. 
This performance variation highlights differences in load distribution practices and enforcement 
effectiveness across the network.
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
This analysis focuses on vehicles exceeding gross vehicle weight limits by more than 2,000 kg. 
A total of {int(total_tagged):,.0f} vehicles were tagged for significant overloading, representing 
{tagged_percentage:.2f}% of total traffic volume across the monitoring network.

{top_tagged['STATION']} reported the highest incidence with {int(top_tagged['GVW_OVERLOADS_FLOAT']):,.0f} 
tagged vehicles, followed by {second_tagged['STATION']} with {int(second_tagged['GVW_OVERLOADS_FLOAT']):,.0f} 
cases. Several stations including {', '.join(low_tagged['STATION'].head(3).tolist()) 
if not low_tagged.empty else 'multiple locations'} demonstrated excellent compliance with 
fewer than 5 tagged vehicles per station.
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
Table 8 presents the distribution of overload incidents by axle configuration. Overall analysis 
reveals that {top_config['AXLE CONFIG']} trucks accounted for the majority of overload cases with 
{top_config['COUNT']:,.0f} incidents ({top_config['PERCENTAGE_VALUE']:.2f}%), indicating this 
configuration as the primary concern for weight compliance enforcement.

{second_config['AXLE CONFIG']} and {third_config['AXLE CONFIG']} configurations also showed 
significant overload activity with {second_config['COUNT']:,.0f} and {third_config['COUNT']:,.0f} 
cases respectively. {moderate_config['AXLE CONFIG']} vehicles contributed moderately with 
{moderate_config['COUNT']:,.0f} overload incidents. Several other configurations including 
{', '.join(other_configs['AXLE CONFIG'].tolist()) if not other_configs.empty else 'various specialty types'} 
demonstrated minimal impact with overload percentages below 2%."""
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
        fig, ax = plt.subplots(figsize=(12, 7))
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
            #label='Known Registration',
            edgecolor='#333333',
            linewidth=0.8
        )
        ax.barh(
            plot_data['Station'],
            plot_data['UNKNOWN_PCT'],
            left=plot_data['KNOWN_PCT'],
            color='#d62728',
            #label='Unknown Registration',
            edgecolor='#333333',
            linewidth=0.8
        )
        
        # Add annotations for known tags (far left inside the bar)
        for i, (station, known_pct, known_count) in enumerate(zip(
                plot_data['Station'],
                plot_data['KNOWN_PCT'],
                plot_data['KNOWN TAGS_NUM']
            )):
            if known_pct > 3:  # Only add annotation if there's enough space
                ax.text(
                    2,  # Far left inside the known portion (2% from start)
                    i,
                    f"{known_count:,.0f} ({known_pct:.1f}%)",
                    va='center',
                    ha='left',
                    fontsize=10,
                    color='white',  # White text for better contrast on green
                    fontweight='bold',
                    bbox=dict(facecolor='none', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.1')
                )
            elif known_pct > 0:  # For very small known portions, place outside
                ax.text(
                    known_pct + 2,  # Just outside the known portion
                    i,
                    f"{known_count:,.0f} ({known_pct:.1f}%)",
                    va='center',
                    ha='left',
                    fontsize=10,
                    color='black',
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3')
                )
        
        # Add annotations for unknown tags (far right inside the bar)
        for i, (station, known_pct, unknown_pct, unknown_count) in enumerate(zip(
                plot_data['Station'],
                plot_data['KNOWN_PCT'],
                plot_data['UNKNOWN_PCT'],
                plot_data['UNKNOWN TAGS_NUM']
            )):
            if unknown_pct > 3:  # Only add annotation if there's enough space
                ax.text(
                    known_pct + unknown_pct - 2,  # Far right inside the unknown portion (2% from end)
                    i,
                    f"{unknown_count:,.0f} ({unknown_pct:.1f}%)",
                    va='center',
                    ha='right',
                    fontsize=10,
                    color='white',  # White text for better contrast on red
                    fontweight='bold',
                    bbox=dict(facecolor='none', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.1')
                )
            elif unknown_pct > 0:  # For very small unknown portions, place outside
                ax.text(
                    known_pct + unknown_pct + 2,  # Just outside the bar
                    i,
                    f"{unknown_count:,.0f} ({unknown_pct:.1f}%)",
                    va='center',
                    ha='left',
                    fontsize=10,
                    color='black',
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3')
                )
        
        ax.set_title('Known Tags Vs Unknown Tags Per Station', fontsize=16, pad=20, weight='bold')
        ax.set_xlabel('Percentage of Vehicles (%)', fontsize=14)
        ax.set_ylabel('Station', fontsize=14)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.set_xlim(0, 100)  # Reset to normal 0-100% scale
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
        #ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor='#333333')
        
        summary_text = (f"Total Unknown: {grand_total['UNKNOWN TAGS'].iloc[0]:,.0f} "
                        f"({grand_total['% UNKNOWN'].iloc[0]:.1f}%) \n"
                        f"Total Vehicles: {grand_total['TOTAL TAGS'].iloc[0]:,.0f}")
        #ax.annotate(summary_text,xy=(0.98, 0.02),xycoords='axes fraction',ha='right',va='bottom',fontsize=10,bbox=dict(facecolor='white', alpha=0.9, edgecolor='#333333', boxstyle='round,pad=0.3'))
        plt.tight_layout()
        sign_off_table = pd.DataFrame({
            ' ': ['PREPARED BY:', 'CHECKED BY:', 'APPROVED BY:'],
            'Name': [' ', ' ', ' '],
            'Sign':[' ', ' ', ' ']
        })
        total_unknown = int(final_table['UNKNOWN TAGS'].iloc[-1].replace(',', ''))
        total_tags = int(final_table['TOTAL TAGS'].iloc[-1].replace(',', ''))
        top_unknown = final_table.iloc[0]
        high_unknown = final_table[final_table['% UNKNOWN'].str.rstrip('%').astype(float) > 50]
        moderate_unknown = final_table[(final_table['% UNKNOWN'].str.rstrip('%').astype(float) >= 20) &
                                    (final_table['% UNKNOWN'].str.rstrip('%').astype(float) <= 50)]
        
        # Use UNKNOWN TAGS_NUM from plot_data for top_unknown's station
        top_unknown_count = plot_data[plot_data['Station'] == top_unknown['Station']]['UNKNOWN TAGS_NUM'].iloc[0]
        analysis_text = f"""
Tagged Unknown Vehicles Analysis: This assessment examines vehicles with unrecognized registrations among overloaded vehicles. A total of {total_unknown:,.0f} unknown tags were identified, representing {grand_total['% UNKNOWN'].iloc[0]:.1f}% of all tagged vehicles. {len(plot_data[plot_data['UNKNOWN TAGS_NUM'] == 0])} stations achieved perfect identification with zero unknown tags, demonstrating optimal registration recognition capabilities.

Primary concern focuses on {top_unknown['Station']} with {top_unknown['UNKNOWN TAGS']} unknown tags accounting for {top_unknown['% UNKNOWN']} of its tagged vehicles. Critical performance issues were noted at {len(high_unknown)} stations with unknown rates exceeding 50%{': ' + ', '.join(high_unknown['Station'].tolist()) if not high_unknown.empty else ''}, while {len(moderate_unknown)} stations showed moderate unknown rates between 20-50%{': ' + ', '.join(moderate_unknown['Station'].tolist()) if not moderate_unknown.empty else ''}."""
        return fig, final_table, sign_off_table, analysis_text