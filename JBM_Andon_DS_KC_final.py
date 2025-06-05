import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import time
import threading
import requests
import sqlite3
from datetime import datetime, date, time as dtime
import json
import tkinter.font as tkfont
import concurrent.futures

CALLTYPES = ['PMD', 'Quality', 'Store', 'JMD', 'Production']
DATABASE_NAME = 'stations04.db'

# FONT SIZE CONSTANTS - Adjust these to change all card fonts at once
FONT_STATION_NAME = 60      # Station name header (was 48)
FONT_CARD_TITLE = 36        # Card titles like "Plan Count", "Actual Count" (was 24)  
FONT_CARD_VALUE_LARGE = 45  # Main values like actual count, plan count (was 36)
FONT_CARD_VALUE_MEDIUM = 38 # Downtime values (was 30)
FONT_TIME_TITLE = 34        # Fault/Resolved time titles (was 20)
FONT_TIME_VALUE = 34        # Fault/Resolved time values (was 24)
FONT_STATUS = 22            # Status messages (was 18)

class DashboardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Andon Signalling System")
        self.configure(bg="white")
        self.minsize(1400, 900)
        # Set a larger default font for the app
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=22)
        self.option_add("*Font", default_font)
        self.option_add("*Menu.Font", default_font)
        self.option_add("*Button.Font", default_font)
        self.option_add("*Label.Font", default_font)
        self.option_add("*Entry.Font", default_font)

        self.station_widgets = {}
        self.station_cards = []
        self.totalStations = []
        self.current_slide_index = 0
        self.auto_slide_id = None
        self.last_actual = 112000
        self.current_shift = get_current_shift()  # Initialize current_shift early
        
        self.logo_image1 = Image.open("./extracted_frames/bad.jpg").resize((200, 120)) or None
        self.logo_photo = ImageTk.PhotoImage(self.logo_image1)
        self.logo_image2 = Image.open("./extracted_frames/bad.jpg").resize((200, 120)) or None
        self.logo_photo1 = ImageTk.PhotoImage(self.logo_image2)

        self.create_menu()
        self.fetch_station_data_from_db()
        self.create_widgets()
        # self.start_auto_slide()
        self.update_station_card()
        self.fetch_data_periodically()
    
    def create_menu(self):
        menubar = tk.Menu(self)
        # Add a File menu as the first menu for macOS compatibility
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Quit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        # Main Options menu
        station_menu = tk.Menu(menubar, tearoff=0)
        station_menu.add_command(label="Add Station", command=self.open_add_station_window)
        station_menu.add_command(label="Edit Station", command=self.open_edit_station_window)
        station_menu.add_command(label="Delete Station", command=self.open_delete_station_window)
        station_menu.add_command(label="Edit Shift Timings", command=self.open_edit_shift_timings_window)
        station_menu.add_separator()
        station_menu.add_command(label="View baydetails", command=lambda: self.view_table('baydetails'))
        station_menu.add_command(label="View SectionData", command=lambda: self.view_table('SectionData'))
        station_menu.add_command(label="View DailyRecord", command=lambda: self.view_table('DailyRecord'))
        station_menu.add_command(label="Shift Data", command=lambda: self.view_table('ShiftData'))
        station_menu.add_command(label="Shift Baselines", command=lambda: self.view_table('ShiftBaselines'))
        station_menu.add_separator()
        station_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="Options", menu=station_menu)
        self.config(menu=menubar)
        self['menu'] = menubar  # For macOS compatibility
        # On macOS, the menu appears at the very top of the screen (system menu bar), not inside the window.

    def view_table(self, table_name):
        view_window = tk.Toplevel(self)
        view_window.title(f"View {table_name}")
        view_window.geometry("1200x600")  # Set a reasonable default size
        view_window.minsize(800, 400)     # Set minimum size

        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cursor.fetchall()]

        # Create a frame for the treeview and scrollbars
        tree_frame = tk.Frame(view_window)
        tree_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Configure style for better row height
        style = ttk.Style()
        style.configure("Custom.Treeview", rowheight=30)  # Increase row height to 30 pixels
        style.configure("Custom.Treeview.Heading", font=('Arial', 12, 'bold'))
        
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15, style="Custom.Treeview")
        
        # Configure column widths based on content
        for col in columns:
            tree.heading(col, text=col)
            # Set reasonable column width based on column name length and content
            if col.lower() in ['id']:
                col_width = 80
            elif col.lower() in ['stationname', 'calltype', 'topic']:
                col_width = 150
            elif col.lower() in ['ipaddress', 'datetime', 'faulttime', 'resolvedtime', 'datecreated']:
                col_width = 180
            elif 'count' in col.lower():
                col_width = 120
            elif 'time' in col.lower():
                col_width = 100
            else:
                col_width = max(120, len(col) * 12)  # Minimum 120px, 12px per character
                
            tree.column(col, width=col_width, anchor='center', minwidth=80)

        # Vertical scrollbar
        v_scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=v_scrollbar.set)
        
        # Horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(tree_frame, orient='horizontal', command=tree.xview)
        tree.configure(xscrollcommand=h_scrollbar.set)

        # Grid the treeview and scrollbars
        tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')

        # Configure grid weights
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        
        for row in rows:
            # Convert None values to empty strings for display
            display_row = [str(item) if item is not None else "" for item in row]
            tree.insert('', tk.END, values=display_row)
        
        conn.close()

        # Add a close button
        close_button = tk.Button(view_window, text="Close", command=view_window.destroy, font=('Arial', 14))
        close_button.pack(pady=5)

    def open_add_station_window(self):
        add_window = tk.Toplevel(self)
        add_window.title("Add Station")
        add_window.geometry("600x400")
        add_window.minsize(500, 350)

        entries = {}
        labels = ["Station Name:", "First Shift Planned Count:", "Second Shift Planned Count:", "Third Shift Planned Count:", "IP Address:", "Topic:"]
        for i, label in enumerate(labels):
            tk.Label(add_window, text=label, font=('Arial', 16)).grid(row=i, column=0, padx=5, pady=5, sticky='e')
            entry = tk.Entry(add_window, font=('Arial', 16), width=25)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky='w')
            entries[label.split(':')[0].lower().replace(' ', '_')] = entry

        def add_station():
            data = {key: entry.get() for key, entry in entries.items()}
            if all(data.values()):
                conn = sqlite3.connect(DATABASE_NAME)
                cursor = conn.cursor()
                try:
                    cursor.execute('''
                        INSERT INTO baydetails (StationName, PlannedCount1, PlannedCount2, PlannedCount3, ipAddress, Topic, isactive, isalive)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (data['station_name'], int(data['first_shift_planned_count']), int(data['second_shift_planned_count']), int(data['third_shift_planned_count']), data['ip_address'], data['topic'], True, True))
                    conn.commit()
                    messagebox.showinfo("Success", f"Station '{data['station_name']}' added successfully.")
                    add_window.destroy()
                    
                    # Refresh data and recreate UI properly with better cleanup
                    self.fetch_station_data_from_db()
                    if self.totalStations > 0:
                        # Clear existing widgets and references properly
                        self.station_widgets.clear()
                        self.station_cards.clear()
                        self.current_slide_index = 0
                        
                        # Recreate all widgets
                        self.create_widgets()
                    self.update_station_card()
                    
                except sqlite3.IntegrityError:
                    messagebox.showerror("Error", f"Station '{data['station_name']}' already exists.")
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numeric values for planned counts.")
                except Exception as e:
                    print(f"Error in add_station: {str(e)}")
                    messagebox.showerror("Error", f"An error occurred: {str(e)}")
                finally:
                    conn.close()
            else:
                messagebox.showerror("Error", "Please fill in all fields.")

        tk.Button(add_window, text="Add Station", command=add_station, font=('Arial', 16)).grid(row=6, column=0, columnspan=2, pady=20)

    def open_edit_station_window(self):
        if not self.stationData:
            messagebox.showinfo("No Stations", "No stations available to edit.")
            return
        # Select station to edit
        select_window = tk.Toplevel(self)
        select_window.title("Select Station to Edit")
        select_window.geometry("400x200")
        select_window.minsize(350, 150)
        
        tk.Label(select_window, text="Select Station:", font=('Arial', 16)).pack(padx=10, pady=10)
        station_var = tk.StringVar(select_window)
        station_var.set(self.stationData[0]['stationName'])
        station_names = [s['stationName'] for s in self.stationData]
        dropdown = ttk.Combobox(select_window, textvariable=station_var, values=station_names, font=('Arial', 16), width=20)
        dropdown.pack(padx=10, pady=10)
        
        def proceed():
            select_window.destroy()
            self._open_edit_station_form(station_var.get())
        
        tk.Button(select_window, text="Edit", command=proceed, font=('Arial', 16)).pack(pady=10)

    def _open_edit_station_form(self, station_name):
        # Fetch station details from DB
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        
        # First try with calltype_index_map column
        try:
            cursor.execute('SELECT PlannedCount1, PlannedCount2, PlannedCount3, ActualCount, ipAddress, Topic, calltype_index_map FROM baydetails WHERE StationName=?', (station_name,))
            row = cursor.fetchone()
            if row:
                planned1, planned2, planned3, actual, ip, topic, calltype_map = row
            else:
                conn.close()
                messagebox.showerror("Error", f"Station '{station_name}' not found.")
                return
        except sqlite3.OperationalError:
            # Column doesn't exist, try without it
            cursor.execute('SELECT PlannedCount1, PlannedCount2, PlannedCount3, ActualCount, ipAddress, Topic FROM baydetails WHERE StationName=?', (station_name,))
            row = cursor.fetchone()
            if row:
                planned1, planned2, planned3, actual, ip, topic = row
                calltype_map = '{"PMD":0,"Quality":2,"Store":6,"JMD":8,"Production":12}'  # Default value
            else:
                conn.close()
                messagebox.showerror("Error", f"Station '{station_name}' not found.")
                return
        
        conn.close()
        
        edit_window = tk.Toplevel(self)
        edit_window.title(f"Edit Station: {station_name}")
        edit_window.geometry("700x600")  # Made window taller for the Text widget
        
        labels = [
            ("First Shift Planned Count:", planned1),
            ("Second Shift Planned Count:", planned2),
            ("Third Shift Planned Count:", planned3),
            ("Actual Count:", actual),
            ("IP Address:", ip),
            ("Topic:", topic)
        ]
        
        entries = {}
        
        # Create regular entry fields
        for i, (label, value) in enumerate(labels):
            tk.Label(edit_window, text=label, font=('Arial', 16)).grid(row=i, column=0, padx=5, pady=5, sticky='e')
            entry = tk.Entry(edit_window, font=('Arial', 16), width=25)
            entry.insert(0, str(value) if value is not None else "")
            entry.grid(row=i, column=1, padx=5, pady=5, sticky='w')
            entries[label] = entry
        
        # Create special Text widget for Calltype Index Map
        tk.Label(edit_window, text="Calltype Index Map (JSON):", font=('Arial', 16)).grid(row=len(labels), column=0, padx=5, pady=5, sticky='ne')
        
        # Create a frame for the text widget and scrollbar
        text_frame = tk.Frame(edit_window)
        text_frame.grid(row=len(labels), column=1, padx=5, pady=5, sticky='w')
        
        calltype_text = tk.Text(text_frame, font=('Arial', 12), width=40, height=4, wrap=tk.WORD)
        calltype_text.insert('1.0', str(calltype_map) if calltype_map is not None else "")
        
        # Add scrollbar for the text widget
        text_scrollbar = tk.Scrollbar(text_frame, orient='vertical', command=calltype_text.yview)
        calltype_text.configure(yscrollcommand=text_scrollbar.set)
        
        calltype_text.pack(side='left', fill='both', expand=True)
        text_scrollbar.pack(side='right', fill='y')
        
        entries["Calltype Index Map (JSON):"] = calltype_text
        
        def save():
            try:
                new_planned1 = int(entries["First Shift Planned Count:"].get())
                new_planned2 = int(entries["Second Shift Planned Count:"].get())
                new_planned3 = int(entries["Third Shift Planned Count:"].get())
                new_actual = int(entries["Actual Count:"].get())
                new_ip = entries["IP Address:"].get()
                new_topic = entries["Topic:"].get()
                new_calltype_map = calltype_text.get('1.0', tk.END).strip()  # Get text from Text widget
                
                # Validate JSON
                json.loads(new_calltype_map)
                
                conn = sqlite3.connect(DATABASE_NAME)
                cursor = conn.cursor()
                
                # Try to update with calltype_index_map column
                try:
                    cursor.execute('''
                        UPDATE baydetails SET PlannedCount1=?, PlannedCount2=?, PlannedCount3=?, ActualCount=?, ipAddress=?, Topic=?, calltype_index_map=? WHERE StationName=?
                    ''', (new_planned1, new_planned2, new_planned3, new_actual, new_ip, new_topic, new_calltype_map, station_name))
                except sqlite3.OperationalError:
                    # Column doesn't exist, update without it
                    cursor.execute('''
                        UPDATE baydetails SET PlannedCount1=?, PlannedCount2=?, PlannedCount3=?, ActualCount=?, ipAddress=?, Topic=? WHERE StationName=?
                    ''', (new_planned1, new_planned2, new_planned3, new_actual, new_ip, new_topic, station_name))
                
                conn.commit()
                conn.close()
                messagebox.showinfo("Success", f"Station '{station_name}' updated successfully.")
                edit_window.destroy()
                
                # Instead of recreating all widgets, just refresh the data
                self.fetch_station_data_from_db()
                self.update_station_card()
                
            except ValueError as e:
                messagebox.showerror("Error", "Please enter valid numeric values for counts.")
            except json.JSONDecodeError:
                messagebox.showerror("Error", "Invalid JSON format for calltype index map.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        
        tk.Button(edit_window, text="Save", command=save, font=('Arial', 16)).grid(row=len(labels)+1, column=0, columnspan=2, pady=20)

    def open_delete_station_window(self):
        if not self.stationData:
            messagebox.showinfo("No Stations", "No stations available to delete.")
            return
            
        # Select station to delete
        delete_window = tk.Toplevel(self)
        delete_window.title("Delete Station")
        delete_window.geometry("500x300")
        delete_window.minsize(400, 250)
        
        # Warning message
        warning_frame = tk.Frame(delete_window, bg="#f8d7da", relief="raised", bd=2)
        warning_frame.pack(fill="x", padx=10, pady=10)
        
        warning_label = tk.Label(
            warning_frame, 
            text="⚠️ WARNING: This will permanently delete the station and all its data!",
            font=('Arial', 14, 'bold'),
            bg="#f8d7da",
            fg="#721c24",
            pady=10
        )
        warning_label.pack()
        
        # Station selection
        selection_frame = tk.Frame(delete_window)
        selection_frame.pack(fill="x", padx=20, pady=20)
        
        tk.Label(selection_frame, text="Select Station to Delete:", font=('Arial', 16, 'bold')).pack(pady=10)
        
        station_var = tk.StringVar(delete_window)
        station_var.set(self.stationData[0]['stationName'])
        station_names = [s['stationName'] for s in self.stationData]
        
        dropdown = ttk.Combobox(
            selection_frame, 
            textvariable=station_var, 
            values=station_names, 
            font=('Arial', 14), 
            width=25,
            state="readonly"
        )
        dropdown.pack(pady=10)
        
        # Station details display
        details_frame = tk.Frame(delete_window, bg="#e9ecef", relief="sunken", bd=2)
        details_frame.pack(fill="x", padx=20, pady=10)
        
        details_label = tk.Label(
            details_frame,
            text="Station details will be shown here...",
            font=('Arial', 12),
            bg="#e9ecef",
            justify="left",
            anchor="w"
        )
        details_label.pack(fill="x", padx=10, pady=10)
        
        def update_station_details(*args):
            selected_station = station_var.get()
            station_info = next((s for s in self.stationData if s['stationName'] == selected_station), None)
            if station_info:
                details_text = f"""Station: {station_info['stationName']}
IP Address: {station_info['ipAddress']}
Planned Count 1: {station_info['planCount']}
Actual Count: {station_info['actualCount']}
Total Downtime: {station_info['totalDowntime']:.1f} mins"""
                details_label.config(text=details_text)
        
        station_var.trace('w', update_station_details)
        update_station_details()  # Initial update
        
        # Buttons
        button_frame = tk.Frame(delete_window)
        button_frame.pack(fill="x", padx=20, pady=20)
        
        def confirm_delete():
            selected_station = station_var.get()
            
            # Double confirmation
            confirm = messagebox.askyesno(
                "Confirm Deletion",
                f"Are you absolutely sure you want to delete station '{selected_station}'?\n\n"
                f"This action will:\n"
                f"• Remove the station from the database\n"
                f"• Delete all associated fault records\n"
                f"• Delete all shift data\n"
                f"• Delete all daily records\n\n"
                f"This action CANNOT be undone!",
                icon="warning"
            )
            
            if confirm:
                try:
                    conn = sqlite3.connect(DATABASE_NAME)
                    cursor = conn.cursor()
                    
                    # Delete from all related tables
                    cursor.execute('DELETE FROM SectionData WHERE StationName=?', (selected_station,))
                    cursor.execute('DELETE FROM DailyRecord WHERE StationName=?', (selected_station,))
                    cursor.execute('DELETE FROM ShiftData WHERE StationName=?', (selected_station,))
                    cursor.execute('DELETE FROM baydetails WHERE StationName=?', (selected_station,))
                    
                    conn.commit()
                    conn.close()
                    
                    messagebox.showinfo("Success", f"Station '{selected_station}' has been deleted successfully.")
                    delete_window.destroy()
                    
                    # Refresh the UI
                    self.fetch_station_data_from_db()
                    if self.totalStations > 0:
                        # Clear existing widgets and references properly
                        self.station_widgets.clear()
                        self.station_cards.clear()
                        self.current_slide_index = 0
                        
                        # Recreate all widgets
                        self.create_widgets()
                    self.update_station_card()
                    
                except Exception as e:
                    print(f"Error deleting station: {str(e)}")
                    messagebox.showerror("Error", f"Failed to delete station: {str(e)}")
                    conn.close()
        
        def cancel_delete():
            delete_window.destroy()
        
        tk.Button(
            button_frame, 
            text="❌ Delete Station", 
            command=confirm_delete, 
            font=('Arial', 14, 'bold'),
            bg="#dc3545",
            fg="white",
            padx=20,
            pady=5
        ).pack(side="left", padx=(0, 10))
        
        tk.Button(
            button_frame, 
            text="Cancel", 
            command=cancel_delete, 
            font=('Arial', 14),
            bg="#6c757d",
            fg="white",
            padx=20,
            pady=5
        ).pack(side="left")

    def open_edit_shift_timings_window(self):
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute('SELECT id, shift1_start, shift1_end, shift2_start, shift2_end, shift3_start, shift3_end FROM ShiftConfig LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        if not row:
            messagebox.showerror("Error", "No shift config found.")
            return
        shift_id, s1_start, s1_end, s2_start, s2_end, s3_start, s3_end = row
        edit_window = tk.Toplevel(self)
        edit_window.title("Edit Shift Timings")
        edit_window.geometry("500x400")
        edit_window.minsize(450, 350)
        
        labels = [
            ("Shift 1 Start (HH:MM):", s1_start),
            ("Shift 1 End (HH:MM):", s1_end),
            ("Shift 2 Start (HH:MM):", s2_start),
            ("Shift 2 End (HH:MM):", s2_end),
            ("Shift 3 Start (HH:MM):", s3_start),
            ("Shift 3 End (HH:MM):", s3_end)
        ]
        entries = {}
        for i, (label, value) in enumerate(labels):
            tk.Label(edit_window, text=label, font=('Arial', 16)).grid(row=i, column=0, padx=5, pady=5, sticky='e')
            entry = tk.Entry(edit_window, font=('Arial', 16), width=20)
            entry.insert(0, str(value) if value is not None else "")
            entry.grid(row=i, column=1, padx=5, pady=5, sticky='w')
            entries[label] = entry
            
        def save():
            try:
                new_s1_start = entries["Shift 1 Start (HH:MM):"].get()
                new_s1_end = entries["Shift 1 End (HH:MM):"].get()
                new_s2_start = entries["Shift 2 Start (HH:MM):"].get()
                new_s2_end = entries["Shift 2 End (HH:MM):"].get()
                new_s3_start = entries["Shift 3 Start (HH:MM):"].get()
                new_s3_end = entries["Shift 3 End (HH:MM):"].get()
                
                # Basic validation for time format
                import re
                time_pattern = r'^([01]?[0-9]|2[0-3]):([0-5][0-9])$'
                times = [new_s1_start, new_s1_end, new_s2_start, new_s2_end, new_s3_start, new_s3_end]
                for time_str in times:
                    if not re.match(time_pattern, time_str):
                        raise ValueError(f"Invalid time format: {time_str}")
                
                conn = sqlite3.connect(DATABASE_NAME)
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE ShiftConfig SET shift1_start=?, shift1_end=?, shift2_start=?, shift2_end=?, shift3_start=?, shift3_end=? WHERE id=?
                ''', (new_s1_start, new_s1_end, new_s2_start, new_s2_end, new_s3_start, new_s3_end, shift_id))
                conn.commit()
                conn.close()
                messagebox.showinfo("Success", "Shift timings updated successfully.")
                edit_window.destroy()
            except ValueError as e:
                messagebox.showerror("Error", f"Please enter valid time format (HH:MM): {str(e)}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                
        tk.Button(edit_window, text="Save", command=save, font=('Arial', 16)).grid(row=len(labels), column=0, columnspan=2, pady=20)

    def create_widgets(self):
        # Clear previous widgets if any with better error handling
        for widget in self.winfo_children():
            if not isinstance(widget, tk.Menu):
                try:
                    widget.destroy()
                except tk.TclError:
                    pass  # Widget already destroyed
        
        # Reset widget collections
        self.station_widgets.clear()
        self.station_cards.clear()
                
        # Set main window background to a pleasant color
        self.configure(bg="#ecf0f1")
        
        header_frame = tk.Frame(self, bg="#ecf0f1")
        header_frame.grid(row=0, column=0, sticky="ew", pady=10)
        header_frame.columnconfigure(2, weight=1)

        tk.Label(header_frame, image=self.logo_photo, bg="white").grid(row=0, column=0, padx=5, pady=5)
        tk.Label(header_frame, image=self.logo_photo1, bg="white").grid(row=0, column=1, padx=0, pady=5)
        tk.Label(header_frame, text="Andon Dashboard", font=("Times New roman", 100, "italic"), bg="#ecf0f1", foreground="blue").grid(row=0, column=2)

        datetime_frame = tk.Frame(header_frame, bg="#ecf0f1")
        datetime_frame.grid(row=0, column=3, padx=10)
        
        # Date label above time label
        self.date_label = tk.Label(datetime_frame, text="", font=("Arial", 65, "bold"), bg="#ecf0f1")
        self.date_label.pack()
        self.header_time_label = tk.Label(datetime_frame, text="", font=("Arial", 65, "bold"), bg="#ecf0f1")
        self.header_time_label.pack()

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Main content frame
        content_frame = tk.Frame(self, bg="#ecf0f1")
        content_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        content_frame.columnconfigure(0, weight=1)  # Station card column takes full width
        content_frame.rowconfigure(0, weight=0)     # Navigation buttons row
        content_frame.rowconfigure(1, weight=1)     # Main content row

        # Navigation buttons frame at the top
        nav_frame = tk.Frame(content_frame, bg="#ecf0f1")
        nav_frame.grid(row=0, column=0, pady=(0, 10), sticky="ew")
        nav_frame.columnconfigure(0, weight=1)
        nav_frame.columnconfigure(1, weight=1)

        # Compact navigation buttons
        self.left_arrow = tk.Button(
            nav_frame, 
            text="◀ Previous", 
            command=self.prev_slide, 
            font=("Arial", 16, "bold"),
            bg="#3498db",
            fg="white",
            relief="raised",
            bd=2,
            padx=20,
            activebackground="#2980b9"
        )
        self.left_arrow.grid(row=0, column=0, padx=(0, 5), sticky="e")

        self.right_arrow = tk.Button(
            nav_frame, 
            text="Next ▶", 
            command=self.next_slide, 
            font=("Arial", 16, "bold"),
            bg="#3498db",
            fg="white",
            relief="raised",
            bd=2,
            padx=20,
            activebackground="#2980b9"
        )
        self.right_arrow.grid(row=0, column=1, padx=(5, 0), sticky="w")

        # Station card container - now takes full width
        self.station_card_frame = tk.Frame(content_frame, bg="#ecf0f1", relief="raised", bd=2)
        self.station_card_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        self.station_card_frame.grid_propagate(False)
        self.station_card_frame.columnconfigure(0, weight=1)
        self.station_card_frame.rowconfigure(0, weight=1)

        # Removed Fault Status Column - no longer needed

        # Create station cards
        print(f"Creating cards for {len(self.stationData)} stations")
        for i, station_info in enumerate(self.stationData):
            print(f"  Creating card for: {station_info['stationName']}")
            station_card = self.create_station_card(self.station_card_frame, station_info)
            self.station_widgets[station_info['stationName']] = station_card
            frame = station_card['frame']
            frame.grid(row=0, column=i, sticky="nsew")
            self.station_cards.append(frame)
       
        # Show first station card if available
        if self.station_cards:
            self.show_station_card(0)
        else:
            print("No station cards to show")

    def create_station_card(self, parent, station_info):
        # Main station card with a modern color scheme
        station_card = tk.Frame(parent, bg="#f0f8ff", padx=20, pady=20)  # Light blue background
        station_card.grid_propagate(False) 
        station_card.grid(row=0, column=0, sticky="nsew")

        station_card.columnconfigure(0, weight=1)
        station_card.columnconfigure(1, weight=1)
        station_card.columnconfigure(2, weight=1)
        station_card.rowconfigure(0, weight=0)  # Station name row
        station_card.rowconfigure(1, weight=1)  # Metrics cards row
        station_card.rowconfigure(2, weight=1)  # Status/fault row

        station_widgets = {"frame": station_card}

        # Station Name Header - Full width
        header_frame = tk.Frame(station_card, bg="#2c3e50", relief="raised", bd=3)
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 20))
        
        station_widgets["station_name_label"] = tk.Label(
            header_frame, 
            text=station_info["stationName"], 
            font=("Arial", FONT_STATION_NAME, "bold"), 
            bg="#2c3e50", 
            fg="white",
            pady=15
        )
        station_widgets["station_name_label"].pack(fill="x")

        # Metrics Cards Row
        metrics_frame = tk.Frame(station_card, bg="#f0f8ff")
        metrics_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(0, 20))
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.columnconfigure(1, weight=1)
        metrics_frame.columnconfigure(2, weight=1)
        metrics_frame.rowconfigure(0, weight=1)

        # Plan Count Card
        plan_card = tk.Frame(metrics_frame, bg="#3498db", relief="raised", bd=3)
        plan_card.grid(row=0, column=0, sticky="nsew", padx=10)
        plan_card.rowconfigure(0, weight=1)
        plan_card.rowconfigure(1, weight=1)
        
        tk.Label(plan_card, text="Plan Count", font=("Arial", FONT_CARD_TITLE, "bold"), bg="#3498db", fg="white").grid(row=0, column=0, pady=(20, 5))
        station_widgets["plan_value"] = tk.Label(plan_card, text=station_info["planCount"], font=("Arial", FONT_CARD_VALUE_LARGE, "bold"), bg="#3498db", fg="white")
        station_widgets["plan_value"].grid(row=1, column=0, pady=(5, 20))

        # Actual Count Card
        actual_card = tk.Frame(metrics_frame, bg="#e74c3c", relief="raised", bd=3)
        actual_card.grid(row=0, column=1, sticky="nsew", padx=10)
        actual_card.rowconfigure(0, weight=1)
        actual_card.rowconfigure(1, weight=1)
        
        tk.Label(actual_card, text="Actual Count", font=("Arial", FONT_CARD_TITLE, "bold"), bg="#e74c3c", fg="white").grid(row=0, column=0, pady=(20, 5))
        station_widgets["actual_value"] = tk.Label(actual_card, text=station_info["actualCount"], font=("Arial", FONT_CARD_VALUE_LARGE, "bold"), bg="#e74c3c", fg="white")
        station_widgets["actual_value"].grid(row=1, column=0, pady=(5, 20))

        # Total Downtime Card
        downtime_card = tk.Frame(metrics_frame, bg="#f39c12", relief="raised", bd=3)
        downtime_card.grid(row=0, column=2, sticky="nsew", padx=10)
        downtime_card.rowconfigure(0, weight=1)
        downtime_card.rowconfigure(1, weight=1)
        
        tk.Label(downtime_card, text="Total Downtime", font=("Arial", FONT_CARD_TITLE, "bold"), bg="#f39c12", fg="white").grid(row=0, column=0, pady=(20, 5))
        station_widgets["total_downtime_value"] = tk.Label(downtime_card, text=f"{station_info['totalDowntime']:.1f} mins", font=("Arial", FONT_CARD_VALUE_LARGE, "bold"), bg="#f39c12", fg="white")
        station_widgets["total_downtime_value"].grid(row=1, column=0, pady=(5, 20))

        # Time Information Row
        time_frame = tk.Frame(station_card, bg="#f0f8ff")
        time_frame.grid(row=2, column=0, columnspan=3, sticky="nsew")
        time_frame.columnconfigure(0, weight=1)
        time_frame.columnconfigure(1, weight=1)
        time_frame.rowconfigure(0, weight=1)

        # Fault Time Card
        fault_time_card = tk.Frame(time_frame, bg="#9b59b6", relief="raised", bd=3)
        fault_time_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        fault_time_card.rowconfigure(0, weight=1)
        fault_time_card.rowconfigure(1, weight=1)
        
        tk.Label(fault_time_card, text="Fault Time", font=("Arial", FONT_TIME_TITLE, "bold"), bg="#9b59b6", fg="white").grid(row=0, column=0, pady=(15, 5))
        station_widgets["fault_time_value"] = tk.Label(fault_time_card, text="", font=("Arial", FONT_TIME_VALUE, "bold"), bg="#9b59b6", fg="white")
        station_widgets["fault_time_value"].grid(row=1, column=0, pady=(5, 15))

        # Resolved Time Card
        resolved_time_card = tk.Frame(time_frame, bg="#1abc9c", relief="raised", bd=3)
        resolved_time_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        resolved_time_card.rowconfigure(0, weight=1)
        resolved_time_card.rowconfigure(1, weight=1)
        
        tk.Label(resolved_time_card, text="Resolved Time", font=("Arial", FONT_TIME_TITLE, "bold"), bg="#1abc9c", fg="white").grid(row=0, column=0, pady=(15, 5))
        station_widgets["resolved_time_value"] = tk.Label(resolved_time_card, text="", font=("Arial", FONT_TIME_VALUE, "bold"), bg="#1abc9c", fg="white")
        station_widgets["resolved_time_value"].grid(row=1, column=0, pady=(5, 15))

        # Status/Calltype Frame (will be updated dynamically)
        station_widgets["calltype_frame"] = None
        station_widgets["calltype_indicators"] = {}

        # Store references to labels that need to be updated (even though they're now part of cards)
        station_widgets["plan_label"] = None  # Not needed anymore but keeping for compatibility
        station_widgets["actual_label"] = None
        station_widgets["total_downtime_label"] = None
        station_widgets["fault_time_label"] = None
        station_widgets["resolved_time_label"] = None
        station_widgets["info_frame"] = metrics_frame  # For background color changes

        return station_widgets

    def show_station_card(self, index):
        # Safely hide all station cards
        for card in self.station_cards:
            try:
                if card.winfo_exists():
                    card.grid_forget()
            except tk.TclError:
                # Widget was destroyed, ignore the error
                pass
        
        self.current_slide_index = index       
        if 0 <= index < len(self.station_cards):
            try:
                if self.station_cards[index].winfo_exists():
                    self.station_cards[index].grid(row=0, column=0, sticky="nsew")
            except tk.TclError:
                # Widget was destroyed, try to recreate widgets
                print("Widget destroyed, recreating...")
                self.create_widgets()
        else:
            print("Index out of range for station cards")

    def update_time(self):
        current_time = time.strftime("%H:%M:%S")
        current_date = time.strftime("%Y-%m-%d")
        # Only update if header_time_label and date_label exist
        if hasattr(self, 'header_time_label') and self.header_time_label.winfo_exists():
            self.header_time_label.config(text=current_time)
        if hasattr(self, 'date_label') and self.date_label.winfo_exists():
            self.date_label.config(text=current_date)
        self.after(1000, self.update_time)
   
    def slide_next_station(self):
        if self.current_slide_index < len(self.station_cards) - 1:
            next_index = self.current_slide_index + 1
        else:
            next_index = 0
        self.current_slide_index = next_index
        self.start_auto_slide()
   
    def start_auto_slide(self):
        self.auto_slide_id = self.after(2000, self.slide_next_station)  # Slide every 2 seconds

    def start_auto_slide(self):
        if self.auto_slide_id is None:
            self.auto_slide_id = self.after(2000, self.next_slide)

    def stop_auto_slide(self):
        if self.auto_slide_id:
            self.after_cancel(self.auto_slide_id)
            self.auto_slide_id = None

    def next_slide(self):
        self.stop_auto_slide()
        self.current_slide_index = (self.current_slide_index + 1) % len(self.station_widgets)
        self.show_station_card(self.current_slide_index)
        self.start_auto_slide()

    def prev_slide(self):
        self.stop_auto_slide()
        self.current_slide_index = (self.current_slide_index - 1) % len(self.station_widgets)
        self.show_station_card(self.current_slide_index)
        self.start_auto_slide()

    def fetch_station_data_from_db(self):
        baydetails = get_baydetails()
        self.stationData = []
        shift_num = get_current_shift()

        for baydetail in baydetails:
            station_name = baydetail['StationName']
            create_shift_data_if_not_exists(station_name)

            shift_Planned = baydetail[f'PlannedCount{shift_num}']
            # Load calltype_index_map from DB (default if missing)
            calltype_index_map = baydetail.get('calltype_index_map')
            if not calltype_index_map:
                calltype_index_map = '{"PMD":0,"Quality":2,"Store":6,"JMD":8,"Production":12}'
            try:
                calltype_index_map = json.loads(calltype_index_map)
            except Exception:
                calltype_index_map = {"PMD":0,"Quality":2,"Store":6,"JMD":8,"Production":12}
            
            # Initialize baseline for current shift if needed
            current_raw_count = baydetail['ActualCount']
            initialize_shift_baseline_if_needed(station_name, shift_num, current_raw_count)
            
            # Calculate shift-relative count for initial display
            shift_relative_count = calculate_shift_relative_count(station_name, current_raw_count, shift_num)
            
            # Start with 0 downtime for current shift (will accumulate as faults occur)
            print(f"📊 Station {station_name}: Shift {shift_num} starting with 0 downtime, Count = {shift_relative_count}")
            
            station_info = {
                'stationName': station_name,
                'prevStates': {ct: 1 for ct in CALLTYPES},  # Initialize all to normal (1)
                'faultData': {ct: {'faultTime': None, 'resolvedTime': None} for ct in CALLTYPES},
                'fault_status': {ct: False for ct in CALLTYPES},  # Initialize all to no fault
                'totalDowntime': 0.0,  # Start with 0 downtime for current shift
                'actualCount': shift_relative_count,  # Display shift-relative count
                'rawActualCount': current_raw_count,  # Store raw count for internal use
                'planCount': shift_Planned,
                'ipAddress': baydetail.get('ipAddress', 'http://192.1.75.191/data'),
                'backgroundColor': 'lightgreen',
                'calltype': '',
                'calltype_index_map': calltype_index_map
            }
            
            print(f"📝 Initialized station {station_name} with fault_status: {station_info['fault_status']}")
            self.stationData.append(station_info)
        self.totalStations = len(self.stationData)

    def update_all_station_cards(self):
        """Update status for all station cards, not just the currently displayed one"""
        for station_info in self.stationData:
            if station_info['stationName'] in self.station_widgets:
                self.update_individual_station_card(station_info)

    def update_individual_station_card(self, station_info):
        """Update a specific station card"""
        if station_info['stationName'] not in self.station_widgets:
            return
            
        station_widgets = self.station_widgets[station_info['stationName']]
        
        # Check if the widgets still exist
        try:
            if not station_widgets["frame"].winfo_exists():
                return
        except tk.TclError:
            return

        is_faulted = any(station_info['fault_status'].values())

        # Update station info with safety checks
        try:
            station_widgets["station_name_label"].config(text=station_info['stationName'])
            station_widgets["plan_value"].config(text=station_info['planCount']) 
            station_widgets["actual_value"].config(text=station_info['actualCount'])
            station_widgets["total_downtime_value"].config(text=f"{station_info['totalDowntime']:.1f} mins")
        except tk.TclError:
            return

        # Update fault/resolved times - show times for any active faults
        fault_time_text = ""
        resolved_time_text = ""
        
        for calltype in CALLTYPES:
            if station_info['fault_status'][calltype]:  # Active fault
                fault_time = station_info['faultData'][calltype]['faultTime']
                if fault_time:
                    fault_time_text = f"{calltype}: {fault_time.strftime('%H:%M:%S')}"
                    break  # Show only the first active fault time
        
        # Show the most recent resolved time
        latest_resolved_time = None
        latest_resolved_calltype = ""
        for calltype in CALLTYPES:
            resolved_time = station_info['faultData'][calltype]['resolvedTime']
            if resolved_time and (latest_resolved_time is None or resolved_time > latest_resolved_time):
                latest_resolved_time = resolved_time
                latest_resolved_calltype = calltype
        
        if latest_resolved_time:
            resolved_time_text = f"{latest_resolved_calltype}: {latest_resolved_time.strftime('%H:%M:%S')}"

        try:
            station_widgets["fault_time_value"].config(text=fault_time_text)
            station_widgets["resolved_time_value"].config(text=resolved_time_text)
        except tk.TclError:
            return

        # Clear existing status frame
        if station_widgets["calltype_frame"]:
            try:
                station_widgets["calltype_frame"].destroy()
                station_widgets["calltype_frame"] = None
                station_widgets["calltype_indicators"] = {}
            except tk.TclError:
                pass

        # Create status indicator for this station
        try:
            if is_faulted:
                station_widgets["calltype_frame"] = tk.Frame(station_widgets["frame"], bg="#e74c3c", relief="raised", bd=3)
                station_widgets["calltype_frame"].grid(row=3, column=0, columnspan=3, sticky="ew", pady=(20, 0))
                
                active_faults = [ct for ct in CALLTYPES if station_info['fault_status'][ct]]
                # Show fault with time if available
                fault_times = []
                for ct in active_faults:
                    fault_time = station_info['faultData'][ct]['faultTime']
                    if fault_time:
                        fault_times.append(f"{ct} ({fault_time.strftime('%H:%M:%S')})")
                    else:
                        fault_times.append(ct)
                
                status_text = f"⚠️ FAULT: {', '.join(fault_times)}"
                
                status_label = tk.Label(
                    station_widgets["calltype_frame"], 
                    text=status_text, 
                    font=('Arial', FONT_STATUS, 'bold'), 
                    bg='#e74c3c', 
                    fg='white',
                    pady=8
                )
                status_label.pack(fill="x")
            else:
                station_widgets["calltype_frame"] = tk.Frame(station_widgets["frame"], bg="#27ae60", relief="raised", bd=3)
                station_widgets["calltype_frame"].grid(row=3, column=0, columnspan=3, sticky="ew", pady=(20, 0))
                
                status_label = tk.Label(
                    station_widgets["calltype_frame"], 
                    text="✅ Operating Normally", 
                    font=('Arial', FONT_STATUS, 'bold'), 
                    bg='#27ae60', 
                    fg='white',
                    pady=8
                )
                status_label.pack(fill="x")
        except tk.TclError:
            pass

    def update_station_card(self):
        if self.totalStations == 0:
            # Clear only content widgets, not the menu
            for widget in self.winfo_children():
                if not isinstance(widget, tk.Menu):
                    widget.destroy()
            
            # Create a simple message frame instead of destroying everything
            no_stations_frame = tk.Frame(self, bg="#ecf0f1")
            no_stations_frame.pack(expand=True, fill="both", padx=20, pady=20)
            
            label = tk.Label(
                no_stations_frame, 
                text="No stations available to display\n\nUse the 'Options' menu to add a new station", 
                font=("Arial", 32), 
                bg="#ecf0f1", 
                fg="#e74c3c",
                justify="center"
            )
            label.pack(expand=True)
            return

        # Safety check to prevent errors when widgets are being recreated
        if (self.current_slide_index >= len(self.stationData) or 
            not hasattr(self, 'station_widgets') or 
            not self.station_widgets):
            return

        # Update all station cards, not just the current one
        self.update_all_station_cards()

        # Removed fault status column update - no longer needed

    def fetch_data_periodically(self):
        # Start background polling thread
        print("🚀 Starting background polling system...")
        threading.Thread(target=self.background_polling_loop, daemon=True).start()
        # Schedule UI updates every 500ms (more responsive)
        self.after(500, self.update_ui_from_polling_data)

    def background_polling_loop(self):
        """Background thread that continuously polls stations"""
        print(f"📡 Background polling thread started for {len(self.stationData)} stations")
        import time
        poll_count = 0
        while True:
            try:
                poll_count += 1
                print(f"\n--- Poll #{poll_count} ---")
                self.poll_all_stations_optimized()
                time.sleep(1.5)  # Poll every 1.5 seconds
            except Exception as e:
                print(f"❌ Error in polling loop: {str(e)}")
                time.sleep(1.5)

    def poll_all_stations_optimized(self):
        """Optimized polling with silent error handling for all stations"""
        
        # Poll all stations (removed 3-station limit)
        stations_to_poll = self.stationData
        
        if not stations_to_poll:
            print("⚠️  No stations to poll")
            return
            
        print(f"🔄 Polling {len(stations_to_poll)} stations...")
        
        def poll_single_station_silent(station_info):
            """Poll a single station with silent error handling"""
            try:
                url = ensure_url_scheme(station_info['ipAddress'])
                print(f"  → Polling {station_info['stationName']} at {url}")
                
                start_time = time.time()
                response = requests.get(url, timeout=5)  # Increased timeout back to 5 seconds
                response_time = time.time() - start_time
                
                if response.status_code == 200 and response.text.strip():
                    # Print successful responses with more detail
                    response_clean = response.text.strip().replace('\n', '').replace('\r', '')
                    print(f"  ✓ {station_info['stationName']}: {response_clean} ({response_time:.2f}s)")
                    return station_info, response.text, True
                else:
                    print(f"  ⚠️  {station_info['stationName']}: Empty or invalid response (Status: {response.status_code})")
                    return station_info, None, False
            except requests.exceptions.Timeout:
                print(f"  ⏱️  {station_info['stationName']}: Connection timeout")
                return station_info, None, False
            except requests.exceptions.ConnectionError:
                print(f"  🔌 {station_info['stationName']}: Connection failed")
                return station_info, None, False
            except Exception as e:
                print(f"  ❌ {station_info['stationName']}: Error - {str(e)}")
                return station_info, None, False

        # Use ThreadPoolExecutor with dynamic workers based on station count
        max_workers = min(len(stations_to_poll), 5)  # Max 5 concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(poll_single_station_silent, station_info): station_info 
                for station_info in stations_to_poll
            }
            
            # Process results quickly without blocking
            timeout_duration = 10  # Increased timeout for more stations
            for future in concurrent.futures.as_completed(futures, timeout=timeout_duration):
                try:
                    station_info, response_text, success = future.result()
                    if success and response_text:
                        # Process in background thread
                        self.process_station_response_optimized(station_info, response_text)
                except concurrent.futures.TimeoutError:
                    print("  ⏱️  Executor timeout occurred")
                except Exception as e:
                    print(f"  ❌ Future processing error: {str(e)}")

    def process_station_response_optimized(self, station_info, response_text):
        """Optimized response processing with better error handling"""
        try:
            print(f"  🔧 Processing data for {station_info['stationName']}")
            
            data_string = response_text.strip('"{}').replace('\n', '').replace('\r', '')
            data_array = list(map(int, data_string.split(',')))
            
            print(f"    Raw data array ({len(data_array)} values): {data_array}")
            
            # Use calltype_index_map for states
            calltype_index_map = station_info.get('calltype_index_map', {"PMD":0,"Quality":2,"Store":6,"JMD":8,"Production":12})
            
            if isinstance(calltype_index_map, str):
                import json
                calltype_index_map = json.loads(calltype_index_map)
            
            print(f"    Calltype map: {calltype_index_map}")
            
            # Get states for each calltype
            states = []
            for ct in CALLTYPES:
                index = calltype_index_map.get(ct, 0)
                if index < len(data_array):
                    states.append(data_array[index])
                    print(f"    {ct} (index {index}): {data_array[index]} ({'FAULT' if data_array[index] == 0 else 'NORMAL'})")
                else:
                    states.append(1)  # Default to normal state
                    print(f"    ⚠️  {ct} index {index} out of range, using default value 1")
            
            # Extract actual count from index 1 (2nd value) - verify this is correct
            actual_count_index = calltype_index_map.get('actual_count', 1)  # Default to index 1 (2nd value)
            
            if actual_count_index < len(data_array):
                actual_count = data_array[actual_count_index]
                print(f"    ✓ Actual count from index {actual_count_index} (value #{actual_count_index + 1}): {actual_count}")
            else:
                actual_count = 0
                print(f"    ⚠️  Actual count index {actual_count_index} out of range, using 0")

            print(f"    Final states: {dict(zip(CALLTYPES, states))}")
            print(f"    Final actual count: {actual_count}")

            # Initialize previous states properly if not exists
            if 'prevStates' not in station_info:
                print(f"    🔄 Initializing previous states for {station_info['stationName']}")
                station_info['prevStates'] = {ct: 1 for ct in CALLTYPES}  # Initialize all to normal (1)
                
            # Store processed data for UI updates
            station_info['_latest_states'] = states
            station_info['_latest_actual_count'] = actual_count
            station_info['_data_updated'] = True

        except Exception as e:
            print(f"    ❌ Error processing {station_info['stationName']}: {str(e)}")
            station_info['_data_updated'] = False

    def update_ui_from_polling_data(self):
        """Update UI from polling data - runs in main thread"""
        try:
            for station_info in self.stationData:
                if station_info.get('_data_updated', False):
                    self.update_station_from_polling_data(station_info)
                    station_info['_data_updated'] = False
        except:
            pass
        
        # Schedule next UI update
        self.after(500, self.update_ui_from_polling_data)

    def update_station_from_polling_data(self, station_info):
        """Update station data from polling results - runs in main thread"""
        try:
            states = station_info.get('_latest_states', [])
            actual_count = station_info.get('_latest_actual_count', 0)
            
            if not states:
                return

            print(f"  🔄 Updating UI for {station_info['stationName']}")
            print(f"    Raw actual count from station: {actual_count}")

            # Calculate difference for database updates
            diff = actual_count - self.last_actual if self.last_actual != 0 else 0
            if diff < 0:
                diff = 0

            now = datetime.now().date()
            prevshift = self.current_shift
            self.current_shift = get_current_shift()
            new_shift1 = new_shift2 = new_shift3 = 0
            
            if self.current_shift == 1:
                new_shift1 = diff
            elif self.current_shift == 2:
                new_shift2 = diff
            elif self.current_shift == 3:
                new_shift3 = diff

            # Handle shift changes and baseline management
            if prevshift != self.current_shift:
                # Set baseline for the new shift using current actual count
                set_shift_baseline(station_info['stationName'], self.current_shift, actual_count, now)
                print(f"🔄 Shift changed from {prevshift} to {self.current_shift} - Set new baseline: {actual_count}")
                
                self.last_actual = actual_count
                
                # Reset downtime to 0 for new shift (fresh start)
                station_info['totalDowntime'] = 0.0
                print(f"    📊 Reset downtime to 0 for new shift {self.current_shift}")
            else:
                # Initialize baseline if it doesn't exist for current shift
                initialize_shift_baseline_if_needed(station_info['stationName'], self.current_shift, actual_count, now)

            # Calculate shift-relative count for display
            shift_relative_count = calculate_shift_relative_count(
                station_info['stationName'], 
                actual_count, 
                self.current_shift, 
                now
            )

            # Store both raw and shift-relative counts
            station_info['actualCount'] = shift_relative_count  # Display shift-relative count in UI
            station_info['rawActualCount'] = actual_count       # Keep raw count for internal use
            station_info['actualCountDiff'] = diff              # Store the calculated difference
            
            print(f"    Raw count: {actual_count}, Shift-relative count: {shift_relative_count} (diff: {diff})")
            
            update_shift_data(station_info['stationName'], actual_count, new_shift1, new_shift2, new_shift3, now)

            # Update UI if station widgets exist - show the shift-relative count
            if station_info['stationName'] in self.station_widgets:
                station_widgets = self.station_widgets[station_info['stationName']]
                try:
                    # Display the shift-relative count (resets at each shift change)
                    station_widgets["actual_value"].config(text=f"{shift_relative_count}")
                    print(f"    ✅ Updated UI actual count display to: {shift_relative_count} (shift-relative)")
                except:
                    print("    ⚠️  Failed to update UI actual count")
                    pass

            # Process fault states - Use dictionary-based approach consistently
            fault_changes_detected = False
            for i, ct in enumerate(CALLTYPES):
                if i < len(states):
                    prev_state = station_info['prevStates'][ct]
                    curr_state = states[i]
                    now_dt = datetime.now()

                    print(f"    {ct}: {prev_state} → {curr_state} ({'No change' if prev_state == curr_state else 'CHANGE DETECTED!'})")

                    if prev_state == 1 and curr_state == 0:  # Fault occurred (1=normal, 0=fault)
                        station_info['fault_status'][ct] = True
                        station_info['faultData'][ct]['faultTime'] = now_dt
                        station_info['faultData'][ct]['resolvedTime'] = None
                        fault_changes_detected = True
                        print(f"🚨 FAULT DETECTED: {station_info['stationName']} - {ct} at {now_dt.strftime('%H:%M:%S')}")

                    elif prev_state == 0 and curr_state == 1:  # Fault resolved (0=fault, 1=normal)
                        station_info['fault_status'][ct] = False
                        fault_time = station_info['faultData'][ct]['faultTime']
                        if fault_time:
                            downtime = (now_dt - fault_time).total_seconds() / 60.0
                            station_info['totalDowntime'] += downtime
                            station_info['faultData'][ct]['resolvedTime'] = now_dt
                            fault_changes_detected = True
                            print(f"✅ FAULT RESOLVED: {station_info['stationName']} - {ct} at {now_dt.strftime('%H:%M:%S')} (Downtime: {downtime:.1f}min)")

                            insert_section_data(
                                station_info['stationName'],
                                ct,
                                fault_time.strftime('%Y-%m-%d %H:%M:%S'),
                                now_dt.strftime('%Y-%m-%d %H:%M:%S'),
                                now_dt.strftime('%Y-%m-%d'),
                                self.current_shift
                            )
                            update_shift_downtime(station_info['stationName'], self.current_shift, station_info['totalDowntime'])
                            update_daily_record(station_info['stationName'], ct, downtime, self.current_shift)
                        station_info['faultData'][ct]['faultTime'] = None
                    
                    elif curr_state == 0:  # Currently in fault state (maintain fault status)
                        station_info['fault_status'][ct] = True
                        if not station_info['faultData'][ct]['faultTime']:
                            # Set fault time if not already set (for existing faults)
                            station_info['faultData'][ct]['faultTime'] = now_dt
                            print(f"⚠️  ONGOING FAULT: {station_info['stationName']} - {ct} (setting fault time)")
                    
                    elif curr_state == 1:  # Currently normal
                        if station_info['fault_status'][ct]:  # Was previously in fault, now resolved
                            station_info['fault_status'][ct] = False
                            station_info['faultData'][ct]['resolvedTime'] = now_dt
                            fault_changes_detected = True
                            print(f"✅ FAULT CLEARED: {station_info['stationName']} - {ct}")

                    # Update the stored previous state for this calltype
                    station_info['prevStates'][ct] = curr_state

            # Force UI update if fault status changed
            if fault_changes_detected:
                print(f"    🔄 Fault changes detected, updating UI for {station_info['stationName']}")
                self.update_individual_station_card(station_info)

            # Always update total downtime display
            if station_info['stationName'] in self.station_widgets:
                station_widgets = self.station_widgets[station_info['stationName']]
                try:
                    station_widgets["total_downtime_value"].config(text=f"{station_info['totalDowntime']:.1f} mins")
                except:
                    pass

        except Exception as e:
            print(f"    ❌ Error updating station {station_info.get('stationName', 'Unknown')}: {str(e)}")
            import traceback
            traceback.print_exc()

    def create_fault_status_column(self, parent):
        # REMOVED - Fault Status Column no longer needed
        pass

    def update_fault_status_column(self):
        # REMOVED - Fault Status Column no longer needed
        pass

    # Keep old methods for compatibility but mark them as deprecated
    def fetch_station_data(self, station_info):
        """Deprecated: Use optimized polling instead"""
        pass

    def fetch_data_thread(self, station_info):
        """Deprecated: Use optimized polling instead"""
        pass

    def process_station_response(self, station_info, response_text):
        """Deprecated: Use process_station_response_optimized instead"""
        pass

    def fetch_all_stations_data(self):
        """Deprecated: Use poll_all_stations_optimized instead"""
        pass

    def set_background_color(self, color, station_name):
        # This method is now simplified since we use individual colored cards
        # instead of changing the entire background color
        if station_name not in self.station_widgets:
            return
            
        station_widgets = self.station_widgets[station_name]
        
        try:
            # Only update the main frame background if needed
            if station_widgets["frame"].winfo_exists():
                # The individual cards maintain their own colors now
                # Main frame stays light blue
                station_widgets["frame"].config(bg="#f0f8ff")
                if "info_frame" in station_widgets and station_widgets["info_frame"]:
                    station_widgets["info_frame"].config(bg="#f0f8ff")
        except tk.TclError:
            # Widget has been destroyed, ignore the error
            pass

def ensure_url_scheme(url):
    """Ensure URL has a proper scheme (http:// or https://)"""
    if not url:
        return 'http://192.1.75.191/data'
    
    # If URL already has a scheme, return as is
    if url.startswith(('http://', 'https://')):
        return url
    
    # If it's just an IP address or domain, add http://
    if not url.startswith('/'):
        return f'http://{url}'
    
    return url

def create_shift_data_if_not_exists(station_name):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM ShiftData WHERE StationName=?', (station_name,))
    record = cursor.fetchone()
    if not record:
        cursor.execute('INSERT INTO ShiftData (StationName) VALUES (?)', (station_name,))
        conn.commit()
    conn.close()

def update_shift_data(station_name, last_actual, shift1_actual, shift2_actual, shift3_actual, now):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE ShiftData SET last_actual_count=?, shift1_actual=?, shift2_actual=?, shift3_actual=?, Date=?
        WHERE StationName=?
    ''', (last_actual, shift1_actual, shift2_actual, shift3_actual, now, station_name))
    conn.commit()
    conn.close()

def get_shift_data(station_name):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT last_actual_count, shift1_actual, shift2_actual, shift3_actual, Date FROM ShiftData WHERE StationName=?', (station_name,))
    record = cursor.fetchone()
    conn.close()
    return {
        'StationName': record[0],
        'last_actual_count': record[1],
        'shift1_actual': record[2],
        'shift2_actual': record[3],
        'shift3_actual': record[4],
        'Date': record[5]
    } if record else {'StationName': "N/A", 'last_actual_count': 0, 'shift1_actual': 0, 'shift2_actual': 0, 'shift3_actual': 0, 'Date': "N/A"}

def create_database():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS baydetails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            StationName TEXT UNIQUE,
            PlannedCount1 INTEGER DEFAULT 0,
            PlannedCount2 INTEGER DEFAULT 0,
            PlannedCount3 INTEGER DEFAULT 0,
            ActualCount INTEGER DEFAULT 0,
            Efficiency REAL DEFAULT 0.00,
            ipAddress TEXT,
            Topic TEXT,
            isactive BOOLEAN DEFAULT 1,
            isalive BOOLEAN DEFAULT 1,
            DateCreated DATETIME DEFAULT CURRENT_TIMESTAMP,
            totalDowntime REAL DEFAULT 0.00,
            calltype_index_map TEXT DEFAULT '{"PMD":0,"Quality":2,"Store":6,"JMD":8,"Production":12}'
        )
    ''')
    
    # Add calltype_index_map column if it doesn't exist (for existing databases)
    try:
        cursor.execute('ALTER TABLE baydetails ADD COLUMN calltype_index_map TEXT DEFAULT \'{"PMD":0,"Quality":2,"Store":6,"JMD":8,"Production":12}\'')
    except sqlite3.OperationalError:
        # Column already exists or other error, continue silently
        pass
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS DailyRecord (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            StationName TEXT,
            TodayDate DATE,
            Plan INTEGER DEFAULT 0,
            ActualCount INTEGER DEFAULT 0,
            Efficiency REAL DEFAULT 0.00,
            mDowntime REAL DEFAULT 0.0,
            pDowntime REAL DEFAULT 0.0,
            qDowntime REAL DEFAULT 0.0,
            sDowntime REAL DEFAULT 0.0,
            jDowntime REAL DEFAULT 0.0,
            totalDowntime REAL DEFAULT 0.0,
            shift INTEGER,
            FOREIGN KEY(StationName) REFERENCES baydetails(StationName)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS SectionData (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            StationName TEXT,
            calltype TEXT,
            FaultTime DATETIME,
            ResolvedTime DATETIME,
            DateTime DATETIME,
            Shift INTEGER,
            FOREIGN KEY(StationName) REFERENCES baydetails(StationName)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ShiftData (
            StationName TEXT UNIQUE,
            last_actual_count INTEGER DEFAULT 0,
            shift1_actual INTEGER DEFAULT 0,
            shift2_actual INTEGER DEFAULT 0,
            shift3_actual INTEGER DEFAULT 0,
            Date DATETIME,
            PRIMARY KEY(StationName)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ShiftConfig (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shift1_start TEXT DEFAULT '05:30',
            shift1_end TEXT DEFAULT '14:20',
            shift2_start TEXT DEFAULT '14:20',
            shift2_end TEXT DEFAULT '00:10',
            shift3_start TEXT DEFAULT '00:10',
            shift3_end TEXT DEFAULT '05:30'
        )
    ''')
    # Insert default if not exists
    cursor.execute('SELECT COUNT(*) FROM ShiftConfig')
    if cursor.fetchone()[0] == 0:
        cursor.execute('''INSERT INTO ShiftConfig (shift1_start, shift1_end, shift2_start, shift2_end, shift3_start, shift3_end) VALUES ('05:30','14:20','14:20','00:10','00:10','05:30')''')
    
    # NEW TABLE: ShiftBaselines to track actual count at start of each shift
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ShiftBaselines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            StationName TEXT,
            Shift INTEGER,
            Date DATE,
            BaselineCount INTEGER DEFAULT 0,
            CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(StationName, Shift, Date),
            FOREIGN KEY(StationName) REFERENCES baydetails(StationName)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_baydetails():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    # First check if calltype_index_map column exists
    cursor.execute("PRAGMA table_info(baydetails)")
    columns = [info[1] for info in cursor.fetchall()]
    has_calltype_map = 'calltype_index_map' in columns
    
    cursor.execute('SELECT * FROM baydetails WHERE isactive=1')
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        station_data = {
            'id': row[0],
            'StationName': row[1],
            'PlannedCount1': row[2],
            'PlannedCount2': row[3],
            'PlannedCount3': row[4],
            'ActualCount': row[5],
            'Efficiency': row[6],
            'ipAddress': row[7],  # Fixed: Use lowercase 'ipAddress' to match database schema
            'Topic': row[8],
            'isactive': row[9],
            'isalive': row[10],
            'DateCreated': row[11] if len(row) > 11 else None,
            'totalDowntime': row[12] if len(row) > 12 else 0.0,
        }
        
        # Add calltype_index_map if column exists
        if has_calltype_map and len(row) > 13:
            station_data['calltype_index_map'] = row[13] if row[13] else '{"PMD":0,"Quality":2,"Store":6,"JMD":8,"Production":12}'
        else:
            station_data['calltype_index_map'] = '{"PMD":0,"Quality":2,"Store":6,"JMD":8,"Production":12}'
        
        result.append(station_data)
    
    return result

def update_total_downtime(station_name, total_downtime):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('UPDATE baydetails SET totalDowntime=? WHERE StationName=?', (total_downtime, station_name))
    conn.commit()
    conn.close()

def update_daily_record(station_name, calltype, downtime, shift):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    today = date.today()
    cursor.execute('SELECT * FROM DailyRecord WHERE StationName=? AND TodayDate=?', (station_name, today))
    record = cursor.fetchone()

    column_map = {
        'Maintenance': 'mDowntime',
        'Production': 'pDowntime',
        'Quality': 'qDowntime',
        'Store': 'sDowntime',
        'JMD': 'jDowntime',
        'PMD': 'mDowntime'
    }

    downtime_col = column_map.get(calltype, 'pDowntime')

    if record:
        cursor.execute(f'UPDATE DailyRecord SET {downtime_col}={downtime_col}+?, totalDowntime=totalDowntime+? WHERE id=?', (downtime, downtime, record[0]))
    else:
        mDowntime = pDowntime = qDowntime = sDowntime = jDowntime = 0.0
        if calltype == 'Maintenance':
            mDowntime = downtime
        elif calltype == 'Production':
            pDowntime = downtime
        elif calltype == 'Quality':
            qDowntime = downtime
        elif calltype == 'Store':
            sDowntime = downtime
        elif calltype == 'JMD':
            jDowntime = downtime
        cursor.execute('''
            INSERT INTO DailyRecord (StationName, TodayDate, mDowntime, pDowntime, qDowntime, sDowntime, jDowntime, totalDowntime, shift)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (station_name, today, mDowntime, pDowntime, qDowntime, sDowntime, jDowntime, downtime, shift))
    conn.commit()
    conn.close()

def insert_section_data(station_name, calltype, fault_time, resolved_time, date, shift):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO SectionData (StationName, calltype, FaultTime, ResolvedTime, DateTime, Shift)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (station_name, calltype, fault_time, resolved_time, date, shift))
    conn.commit()
    conn.close()

def get_current_shift():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT shift1_start, shift1_end, shift2_start, shift2_end, shift3_start, shift3_end FROM ShiftConfig LIMIT 1')
    row = cursor.fetchone()
    conn.close()
    now = datetime.now().time()
    def parse_time(tstr):
        h, m = map(int, tstr.split(':'))
        return dtime(h, m)
    if row:
        s1_start, s1_end, s2_start, s2_end, s3_start, s3_end = row
        shift1_start = parse_time(s1_start)
        shift1_end = parse_time(s1_end)
        shift2_start = parse_time(s2_start)
        shift2_end = parse_time(s2_end)
        shift3_start = parse_time(s3_start)
        shift3_end = parse_time(s3_end)
        if shift1_start <= now < shift1_end:
            return 1
        elif shift2_start <= now < shift2_end:
            return 2
        else:
            return 3
    # fallback
    if dtime(5, 30) <= now < dtime(14, 20):
        return 1
    elif dtime(14, 20) <= now < dtime(0, 10):
        return 2
    return 3

def get_shift_downtime(station_name, shift_num):
    """Get the downtime for a specific station and shift"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    today = date.today()
    
    cursor.execute('SELECT totalDowntime FROM DailyRecord WHERE StationName=? AND TodayDate=? AND shift=?', 
                   (station_name, today, shift_num))
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result else 0.0

def update_shift_downtime(station_name, shift_num, total_downtime):
    """Update the downtime for a specific station and shift"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    today = date.today()
    
    # Check if record exists
    cursor.execute('SELECT id FROM DailyRecord WHERE StationName=? AND TodayDate=? AND shift=?', 
                   (station_name, today, shift_num))
    result = cursor.fetchone()
    
    if result:
        # Update existing record
        cursor.execute('UPDATE DailyRecord SET totalDowntime=? WHERE id=?', 
                       (total_downtime, result[0]))
    else:
        # Create new record
        cursor.execute('''INSERT INTO DailyRecord (StationName, TodayDate, shift, totalDowntime) 
                         VALUES (?, ?, ?, ?)''', 
                       (station_name, today, shift_num, total_downtime))
    
    conn.commit()
    conn.close()

def reset_shift_downtime(station_name, shift_num):
    """Reset downtime for a specific station and shift to 0"""
    update_shift_downtime(station_name, shift_num, 0.0)

# NEW FUNCTIONS: Shift Baseline Management
def get_shift_baseline(station_name, shift_num, date_obj=None):
    """Get the baseline count for a specific station, shift, and date"""
    if date_obj is None:
        date_obj = date.today()
    
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT BaselineCount FROM ShiftBaselines 
        WHERE StationName=? AND Shift=? AND Date=?
    ''', (station_name, shift_num, date_obj))
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result else None

def set_shift_baseline(station_name, shift_num, baseline_count, date_obj=None):
    """Set the baseline count for a specific station, shift, and date"""
    if date_obj is None:
        date_obj = date.today()
    
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    # Use INSERT OR REPLACE to handle existing records
    cursor.execute('''
        INSERT OR REPLACE INTO ShiftBaselines (StationName, Shift, Date, BaselineCount, CreatedAt)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (station_name, shift_num, date_obj, baseline_count))
    
    conn.commit()
    conn.close()

def calculate_shift_relative_count(station_name, current_actual_count, shift_num, date_obj=None):
    """Calculate shift-relative count by subtracting baseline from current count"""
    baseline = get_shift_baseline(station_name, shift_num, date_obj)
    
    if baseline is None:
        # No baseline exists, set current count as baseline and return 0
        set_shift_baseline(station_name, shift_num, current_actual_count, date_obj)
        return 0
    
    # Return shift-relative count (ensure it's not negative)
    shift_count = current_actual_count - baseline
    return max(0, shift_count)

def initialize_shift_baseline_if_needed(station_name, shift_num, current_actual_count, date_obj=None):
    """Initialize baseline for current shift if it doesn't exist"""
    if date_obj is None:
        date_obj = date.today()
    
    baseline = get_shift_baseline(station_name, shift_num, date_obj)
    if baseline is None:
        set_shift_baseline(station_name, shift_num, current_actual_count, date_obj)
        print(f"🔄 Initialized baseline for {station_name} Shift {shift_num}: {current_actual_count}")
        return True
    return False

if __name__ == "__main__":
    create_database()
    app = DashboardApp()
    app.update_time()
    app.mainloop()
