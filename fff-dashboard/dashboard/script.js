// FFF Fitness Studio Dashboard
class FFFDashboard {
    constructor() {
        this.currentStudio = null;
        this.currentDate = new Date().toISOString().split('T')[0];
        this.chart = null;
        this.updateInterval = null;
        this.studios = [];
        
        this.init();
    }

    async init() {
        await this.loadStudios();
        this.setupEventListeners();
        this.setupTheme();
        this.setupDatePicker();
        this.startAutoUpdate();
        
        // Restore last viewed studio and date
        this.restoreLastState();
        
        // Load initial data if studio is selected
        if (this.currentStudio) {
            await this.loadData();
        }
    }

    async loadStudios() {
        try {
            // Load available studios from your API
            const response = await fetch('http://localhost:8000/api/v1/gyms');
            if (response.ok) {
                this.studios = await response.json();
                this.populateStudioSelector();
            } else {
                // Fallback to known studios if API is not available
                this.studios = [
                    { uuid: '1b793462-e413-49fb-b971-ada1e11dc90e', name: 'Darmstadt - Ostbahnhof' }
                ];
                this.populateStudioSelector();
            }
        } catch (error) {
            console.warn('Could not load studios from API, using fallback:', error);
            this.studios = [
                { uuid: '1b793462-e413-49fb-b971-ada1e11dc90e', name: 'Darmstadt - Ostbahnhof' }
            ];
            this.populateStudioSelector();
        }
    }

    populateStudioSelector() {
        const selector = document.getElementById('studioSelector');
        selector.innerHTML = '<option value="">Studio auswählen...</option>';
        
        this.studios.forEach(studio => {
            const option = document.createElement('option');
            option.value = studio.uuid;
            option.textContent = studio.name;
            selector.appendChild(option);
        });
    }

    setupEventListeners() {
        // Studio selector change
        document.getElementById('studioSelector').addEventListener('change', (e) => {
            this.currentStudio = e.target.value;
            if (this.currentStudio) {
                // Save selected studio
                localStorage.setItem('fff_last_studio', this.currentStudio);
                this.updateWorkingHoursDisplay(); // Update working hours when studio changes
                this.loadData();
            } else {
                this.hideDashboard();
            }
        });

        // Date picker change
        document.getElementById('datePicker').addEventListener('change', (e) => {
            this.currentDate = e.target.value;
            // Save selected date
            localStorage.setItem('fff_last_date', this.currentDate);
            this.updateWorkingHoursDisplay(); // Update working hours when date changes
            if (this.currentStudio) {
                this.loadData();
            }
        });
        
        // Date navigation buttons
        document.getElementById('prevDayBtn').addEventListener('click', () => this.navigateDate(-1));
        document.getElementById('nextDayBtn').addEventListener('click', () => this.navigateDate(1));

        // Theme toggle
        document.getElementById('themeToggle').addEventListener('click', () => {
            this.toggleTheme();
        });


    }

    setupTheme() {
        // Check for saved theme preference or default to dark mode
        const savedTheme = localStorage.getItem('fff_theme') || 'dark';
        this.setTheme(savedTheme);
    }

    restoreLastState() {
        // Restore last viewed studio
        const savedStudio = localStorage.getItem('fff_last_studio');
        if (savedStudio && this.studios.find(s => s.uuid === savedStudio)) {
            this.currentStudio = savedStudio;
            document.getElementById('studioSelector').value = savedStudio;
        }
        
        // Restore last viewed date (default to today if not set or invalid)
        const savedDate = localStorage.getItem('fff_last_date');
        if (savedDate && this.isValidDate(savedDate)) {
            this.currentDate = savedDate;
            document.getElementById('datePicker').value = savedDate;
        }
        
        // Update working hours display if studio is restored
        if (this.currentStudio) {
            this.updateWorkingHoursDisplay();
        }
    }

    isValidDate(dateString) {
        const date = new Date(dateString);
        return date instanceof Date && !isNaN(date);
    }

    setTheme(theme) {
        if (theme === 'dark') {
            document.documentElement.classList.add('dark');
            document.getElementById('sunIcon').classList.remove('hidden');
            document.getElementById('moonIcon').classList.add('hidden');
        } else {
            document.documentElement.classList.remove('dark');
            document.getElementById('sunIcon').classList.add('hidden');
            document.getElementById('moonIcon').classList.remove('hidden');
        }
        localStorage.setItem('fff_theme', theme);
        
        // Chart neu erstellen, wenn es existiert und Daten geladen sind
        if (this.chart && this.currentStudio && this.currentDate) {
            this.recreateChart();
        }
    }

    toggleTheme() {
        const isDark = document.documentElement.classList.contains('dark');
        this.setTheme(isDark ? 'light' : 'dark');
    }

    setupDatePicker() {
        const datePicker = document.getElementById('datePicker');
        datePicker.value = this.currentDate;
        
        // Allow selection up to 2 days in the future (for forecast data)
        const maxDate = new Date();
        maxDate.setDate(maxDate.getDate() + 2);
        datePicker.max = maxDate.toISOString().split('T')[0];
        
        // Update navigation button states
        this.updateNavigationButtons();
    }




    
    navigateDate(direction) {
        const currentDate = new Date(this.currentDate);
        currentDate.setDate(currentDate.getDate() + direction);
        
        // Format as YYYY-MM-DD
        this.currentDate = currentDate.toISOString().split('T')[0];
        
        // Update date picker
        document.getElementById('datePicker').value = this.currentDate;
        
        // Save selected date
        localStorage.setItem('fff_last_date', this.currentDate);
        
        // Update navigation button states
        this.updateNavigationButtons();
        
        // Update working hours for new date
        this.updateWorkingHoursDisplay();
        
        // Load data for new date
        if (this.currentStudio) {
            this.loadData();
        }
    }
    
    updateNavigationButtons() {
        const currentDate = new Date(this.currentDate);
        const today = new Date();
        const maxDate = new Date();
        maxDate.setDate(maxDate.getDate() + 2); // Allow up to 2 days in future
        
        // Previous day button - always enabled
        const prevBtn = document.getElementById('prevDayBtn');
        prevBtn.disabled = false;
        prevBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        
        // Next day button - disabled if current date is 2 days from today or later
        const nextBtn = document.getElementById('nextDayBtn');
        if (currentDate >= maxDate) {
            nextBtn.disabled = true;
            nextBtn.classList.add('opacity-50', 'cursor-not-allowed');
        } else {
            nextBtn.disabled = false;
            nextBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    }

    async loadData() {
        if (!this.currentStudio || !this.currentDate) return;

        this.showLoading();
        this.hideError();

        try {
            const timeseriesData = await this.fetchTimeseriesData();

            // Extract mean and forecast data from timeseries response (new API structure)
            const meanData = timeseriesData && timeseriesData.mean ? timeseriesData.mean : null;
            const forecastData = timeseriesData && timeseriesData.forecast ? timeseriesData.forecast : null;

            this.processData(meanData, forecastData);
            this.showDashboard();
        } catch (error) {
            console.error('Error loading data:', error);
            this.showError('Fehler beim Laden der Daten: ' + error.message);
            this.hideDashboard();
        }
    }

    async fetchTimeseriesData() {
        const response = await fetch(`http://localhost:8000/api/v1/gyms/${this.currentStudio}/timeseries?date=${this.currentDate}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    }

    async fetchForecastData() {
        try {
            // Forecast data is included in the timeseries API response
            // We'll extract it from the timeseries data in processData
            return null; // No separate API call needed
        } catch (error) {
    
            return null;
        }
    }




    processData(meanData, forecastData) {
        // Process mean data (actual utilization)
        const processedData = this.processTimeseriesData(meanData);
        
        // Process forecast data
        const processedForecast = this.processTimeseriesData(forecastData);
        
        // Store processed data for chart recreation
        this.lastProcessedData = {
            timeseries: processedData,
            forecast: processedForecast
        };
        
        // Update overview cards
        this.updateOverviewCards(processedData);
        
        // Create/update chart
        this.createChart(processedData, processedForecast);
    }

    processTimeseriesData(data) {
        if (!data) return [];
        
        // Process data with consistent format
        const processedData = data.map(item => {
            const timestamp = new Date(item.timestamp);
            
            return {
                timestamp: timestamp,
                value: parseFloat(item.value) || 0,
                time: timestamp.toLocaleTimeString('de-DE', { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                })
            };
        });
        

        return processedData.sort((a, b) => a.timestamp - b.timestamp);
    }

    updateOverviewCards(data) {
        
        // Current date
        document.getElementById('currentDate').textContent = new Date(this.currentDate).toLocaleDateString('de-DE', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
        
        if (data.length === 0) return;

        // Average utilization
        const avgValue = data.reduce((sum, item) => sum + item.value, 0) / data.length;
        document.getElementById('avgUtilization').textContent = `${avgValue.toFixed(1)} Personen`;

        // Peak time with person count
        const peakItem = data.reduce((max, item) => item.value > max.value ? item : max, data[0]);
        document.getElementById('peakTime').textContent = `${peakItem.time} (${peakItem.value.toFixed(0)} Personen)`;

        // Working hours - get from current studio data
        const workingHours = this.getCurrentWorkingHours();
        document.getElementById('workingHours').textContent = workingHours;
    }

    getCurrentWorkingHours() {
        // Get current studio information
        const currentStudioData = this.studios.find(studio => studio.uuid === this.currentStudio);
        if (!currentStudioData || !currentStudioData.working_hours) {
            return '6:00 - 22:00'; // Fallback
        }
        
        // Get current day name (3-letter format like "Mon", "Tue", etc.)
        const currentDate = new Date(this.currentDate);
        const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        const currentDay = dayNames[currentDate.getDay()];
        

        
        // Get working hours for current day
        const workingHours = currentStudioData.working_hours[currentDay];
        if (!workingHours) {
            return 'Geschlossen'; // Closed
        }
        
        // Format working hours string (e.g., "06:00-22:00" -> "6:00 - 22:00")
        if (workingHours.includes('-')) {
            const [openTime, closeTime] = workingHours.split('-');
            // Remove leading zeros and add spaces around dash
            const formattedOpen = openTime.replace(/^0/, '');
            const formattedClose = closeTime.replace(/^0/, '');
            return `${formattedOpen} - ${formattedClose}`;
        }
        
        return workingHours;
    }

    updateWorkingHoursDisplay() {
        // Update the working hours display in the UI
        const workingHours = this.getCurrentWorkingHours();
        const workingHoursElement = document.getElementById('workingHours');
        if (workingHoursElement) {
            workingHoursElement.textContent = workingHours;
        }
    }





    hasValidForecastData(forecastData) {
        // Check if forecast data exists and is meaningful
        if (!forecastData || !Array.isArray(forecastData) || forecastData.length === 0) {
            return false;
        }
        
        // Check if forecast has any non-zero values (indicating actual forecast data)
        const hasNonZeroValues = forecastData.some(item => item.value > 0);
        
        // Check if forecast data covers a reasonable time range (at least 2 hours)
        const timeRange = forecastData.length > 0 ? 
            (forecastData[forecastData.length - 1].timestamp - forecastData[0].timestamp) / (1000 * 60 * 60) : 0;
        
        return hasNonZeroValues && timeRange >= 2; // At least 2 hours of data
    }

    getOpeningHoursRange() {
        // Get current studio working hours
        const currentStudio = this.studios.find(studio => studio.uuid === this.currentStudio);
        if (!currentStudio || !currentStudio.working_hours) {
            return { openHour: 0, closeHour: 24 }; // Default to full day if no working hours defined
        }

        // Get current date to determine day of week
        const currentDate = new Date(this.currentDate);
        const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        const currentDay = dayNames[currentDate.getDay()];
        
        // Get working hours for current day
        const workingHours = currentStudio.working_hours[currentDay];
        if (!workingHours || !workingHours.includes('-')) {
            return { openHour: 0, closeHour: 24 }; // Default to full day if no valid working hours
        }

        // Parse working hours (e.g., "06:00-22:00")
        const [openStr, closeStr] = workingHours.split('-');
        const openHour = parseInt(openStr.split(':')[0]);
        const closeHour = parseInt(closeStr.split(':')[0]);

        return { openHour, closeHour };
    }

    generateTimeLabels(openHour, closeHour) {
        const labels = [];
        for (let hour = openHour; hour <= closeHour; hour++) {
            for (let minute = 0; minute < 60; minute += 10) {
                // Skip the last iteration if we're at the closing hour and minute > 0
                if (hour === closeHour && minute > 0) {
                    break;
                }
                labels.push(`${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`);
            }
        }
        return labels;
    }

    syncDataWithLabels(data, labels) {
        const syncedData = [];
        
        labels.forEach((label, index) => {
            // Find matching data point
            const dataPoint = data.find(item => item.time === label);
            if (dataPoint) {
                syncedData.push(dataPoint.value);
            } else {
                // Interpolate between available data points
                const interpolatedValue = this.interpolateValue(labels, data, index);
                syncedData.push(interpolatedValue);
            }
        });
        
        return syncedData;
    }

    interpolateValue(labels, data, currentIndex) {
        // Find the nearest data points before and after the current time
        let beforeIndex = -1;
        let afterIndex = -1;
        
        // Look for data point before current time
        for (let i = currentIndex - 1; i >= 0; i--) {
            const dataPoint = data.find(item => item.time === labels[i]);
            if (dataPoint) {
                beforeIndex = i;
                break;
            }
        }
        
        // Look for data point after current time
        for (let i = currentIndex + 1; i < labels.length; i++) {
            const dataPoint = data.find(item => item.time === labels[i]);
            if (dataPoint) {
                afterIndex = i;
                break;
            }
        }
        
        // Only interpolate if we have BOTH before and after data points
        // This ensures we only interpolate between existing data, not before or after
        if (beforeIndex !== -1 && afterIndex !== -1) {
            const beforeData = data.find(item => item.time === labels[beforeIndex]);
            const afterData = data.find(item => item.time === labels[afterIndex]);
            
            // Linear interpolation: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            const x1 = beforeIndex;
            const x2 = afterIndex;
            const y1 = beforeData.value;
            const y2 = afterData.value;
            const x = currentIndex;
            
            const interpolatedValue = y1 + (x - x1) * (y2 - y1) / (x2 - x1);
            return Math.max(0, Math.round(interpolatedValue * 100) / 100); // Ensure non-negative and round to 2 decimals
        }
        
        // If we don't have both data points, return null to create gaps
        // This prevents interpolation before the first or after the last data point
        return null;
    }

    filterDataByOpeningHours(data) {
        if (!data || data.length === 0) {
            return [];
        }

        const { openHour, closeHour } = this.getOpeningHoursRange();

        // Filter data to only show times within opening hours
        return data.filter(item => {
            // Parse the time string (e.g., "14:30")
            const timeMatch = item.time.match(/(\d{1,2}):(\d{2})/);
            if (!timeMatch) {
                return true; // Include if we can't parse the time
            }
            
            const hour = parseInt(timeMatch[1]);
            const minute = parseInt(timeMatch[2]);
            
            // Convert to decimal hours for easier comparison
            const decimalHour = hour + (minute / 60);
            
            // Check if time is within opening hours
            // Studio is open from openHour (inclusive) to closeHour (inclusive)
            return decimalHour >= openHour && decimalHour <= closeHour;
        });
    }

    createChart(timeseriesData, forecastData) {
        const ctx = document.getElementById('mainChart').getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }



                // Get opening hours range and generate consistent time labels
        const { openHour, closeHour } = this.getOpeningHoursRange();
        const labels = this.generateTimeLabels(openHour, closeHour);

        // Filter data to show only within opening hours
        const filteredTimeseriesData = this.filterDataByOpeningHours(timeseriesData);
        const filteredForecastData = this.filterDataByOpeningHours(forecastData);

        const chartData = {
            labels: labels,
            datasets: []
        };

        // Add timeseries data if available
        if (filteredTimeseriesData.length > 0) {
            const timeseriesData = this.syncDataWithLabels(filteredTimeseriesData, labels);
            chartData.datasets.push({
                label: 'Tatsächliche Auslastung',
                data: timeseriesData,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            });
        }

        // Add forecast data if available
        if (filteredForecastData && filteredForecastData.length > 0) {
            const forecastData = this.syncDataWithLabels(filteredForecastData, labels);
            chartData.datasets.push({
                label: 'Vorhersage',
                data: forecastData,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderDash: [5, 5],
                fill: filteredTimeseriesData.length === 0, // Fill if no historical data
                tension: 0.4
            });
        }

        // Force text colors for chart
        const isDarkMode = document.documentElement.classList.contains('dark');
        const textColor = isDarkMode ? '#ffffff' : '#111827';

        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },

                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: textColor,
                            font: {
                                size: 16,
                                weight: 'bold'
                            },
                            usePointStyle: true,
                            pointStyle: 'circle',
                            padding: 20,
                            // Force text color with multiple properties
                            textStrokeColor: textColor,
                            textStrokeWidth: 0,
                            fillStyle: textColor
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.95)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#3b82f6',
                        borderWidth: 2,
                        titleFont: {
                            size: 16,
                            weight: 'bold'
                        },
                        bodyFont: {
                            size: 14,
                            weight: '500'
                        },
                        padding: 12,
                        cornerRadius: 8
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Zeit',
                            color: textColor,
                            font: {
                                size: 16,
                                weight: 'bold'
                            },
                            padding: 10
                        },
                        ticks: {
                            color: isDarkMode ? '#f3f4f6' : '#6b7280',
                            font: {
                                size: 13,
                                weight: '500'
                            },
                            maxTicksLimit: (closeHour - openHour + 1), // Dynamisch basierend auf Öffnungszeiten
                            autoSkip: true, // Überspringe automatisch Ticks
                            autoSkipPadding: 5 // Abstand zwischen sichtbaren Ticks
                        },
                        grid: {
                            color: isDarkMode ? '#6b7280' : '#e5e7eb',
                            lineWidth: 1
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Auslastung (Personen)',
                            color: textColor,
                            font: {
                                size: 16,
                                weight: 'bold'
                            },
                            padding: 10
                        },
                        ticks: {
                            color: isDarkMode ? '#f3f4f6' : '#6b7280',
                            font: {
                                size: 13,
                                weight: '500'
                            },
                            callback: function(value) {
                                return value + ' Pers.';
                            }
                        },
                        grid: {
                            color: isDarkMode ? '#6b7280' : '#e5e7eb',
                            lineWidth: 1
                        },
                        min: 0,
                        max: Math.max(
                            ...(filteredTimeseriesData.length > 0 ? filteredTimeseriesData.map(item => item.value) : [0]), 
                            ...(filteredForecastData.length > 0 ? filteredForecastData.map(item => item.value) : [0])
                        ) + 5
                    }
                }
            }
        });
    }

    recreateChart() {
        // Bestehenden Chart zerstören
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }
        
        // Chart mit den aktuellen Daten neu erstellen (ohne Daten neu zu laden)
        if (this.currentStudio && this.currentDate && this.lastProcessedData) {
            this.createChart(this.lastProcessedData.timeseries, this.lastProcessedData.forecast);
        }
    }

    showLoading() {
        document.getElementById('loadingState').classList.remove('hidden');
        document.getElementById('dashboardContent').classList.add('hidden');
    }

    hideLoading() {
        document.getElementById('loadingState').classList.add('hidden');
    }

    showDashboard() {
        this.hideLoading();
        document.getElementById('dashboardContent').classList.remove('hidden');
        document.getElementById('dashboardContent').classList.add('fade-in-up');
    }

    hideDashboard() {
        document.getElementById('dashboardContent').classList.add('hidden');
        document.getElementById('dashboardContent').classList.remove('fade-in-up');
    }

    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('errorState').classList.remove('hidden');
    }

    hideError() {
        document.getElementById('errorState').classList.add('hidden');
    }

    startAutoUpdate() {
        // Update data every 5 minutes
        this.updateInterval = setInterval(() => {
            if (this.currentStudio) {
                this.loadData();
            }
        }, 5 * 60 * 1000);
    }

    stopAutoUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FFFDashboard();
});

// Handle page visibility changes to pause updates when tab is not visible
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Pause updates when tab is not visible
        if (window.fffDashboard) {
            window.fffDashboard.stopAutoUpdate();
        }
    } else {
        // Resume updates when tab becomes visible
        if (window.fffDashboard) {
            window.fffDashboard.startAutoUpdate();
        }
    }
});
