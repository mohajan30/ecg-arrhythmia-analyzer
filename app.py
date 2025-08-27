# File upload
st.subheader("Upload ECG Files")
dat_file = st.file_uploader("Upload .dat file", type=["dat"])
atr_file = st.file_uploader("Upload .atr file", type=["atr"])

if dat_file and atr_file:
    record_name = dat_file.name.split(".")[0]
    if record_name != atr_file.name.split(".")[0]:
        st.error("Uploaded .dat and .atr files must have the same name.")
    else:
        # Save uploaded files
        with open(f"{record_name}.dat", "wb") as f: f.write(dat_file.read())
        with open(f"{record_name}.atr", "wb") as f: f.write(atr_file.read())

        # Burden Analysis
        st.subheader("Burden Analysis")
        counts, total_beats = load_record_counts(record_name)
        if total_beats == 0:
            st.error("No valid beats found in the uploaded record.")
        else:
            duration_hours = 0.5
            rows = []
            for k in AAMI10:
                c = counts[k]
                p = 100 * c / total_beats if total_beats>0 else 0
                h = c / duration_hours
                zone = flag_color(k, p, c, h)
                rows.append([k, c, f"{p:.2f} %", str(complication.get(k,"N/A")), zone])
            df_burden = pd.DataFrame(rows, columns=["Class", "# Beats", "%", "Possible Complication", "Zone"])
            def hilite_zone(row):
                color_map = {"Safe":"#d4edda", "Elevated":"#fff3cd", "Dangerous":"#f8d7da"}
                return [f'background-color: {color_map.get(row.Zone,"white")}']*len(row)
            st.dataframe(df_burden.style.apply(hilite_zone, axis=1))
            st.download_button("Download Burden Table", df_burden.to_csv(index=False), f"burden_{record_name}.csv")

        # Disease Risk Assessment
        st.subheader("Disease Risk Assessment")
        disease_rows=[]
        for disease in disease_mappings:
            zone = assign_disease_zone(disease, counts, total_beats, duration_hours)
            markers = disease_mappings[disease]["yellow"] if zone=="Elevated" else disease_mappings[disease]["red"] if zone=="Dangerous" else "None"
            comps = disease_mappings[disease]["comps"] if zone!="Safe" else "None triggered"
            disease_rows.append([disease, markers, zone, comps])
        df_diseases = pd.DataFrame(disease_rows, columns=["Disease","Markers & Thresholds Met","Zone","Complications"])
        st.dataframe(df_diseases.style.apply(hilite_zone, axis=1))
        st.download_button("Download Disease Table", df_diseases.to_csv(index=False), f"diseases_{record_name}.csv")

        # Cleanup
        try:
            os.remove(f"{record_name}.dat")
            os.remove(f"{record_name}.atr")
        except: pass

else:
    st.write("Please upload both .dat and .atr files to proceed.")
