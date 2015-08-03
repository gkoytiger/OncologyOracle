SELECT data.assays.assay_id, data.assays.doc_id, data.assays.assay_cell_type, data.assays.cell_id,  
data.activities.molregno, data.activities.standard_value, data.activities.standard_units
FROM data.assays
JOIN data.activities
ON data.assays.assay_id = data.activities.assay_id
WHERE data.assays.assay_organism = 'Homo sapiens' AND data.assays.assay_type = 'F'AND data.assays.cell_id IS NOT NULL 
AND data.activities.data_validity_comment is NULL AND data.activities.standard_value is not NULL AND data.activities.standard_type = 'IC50' AND data.activities.standard_units = 'nM';




CREATE TABLE data.temp AS( 
SELECT data.assays.assay_id, data.assays.doc_id, data.assays.assay_cell_type, data.assays.cell_id,  
data.activities.molregno, data.activities.standard_value, data.activities.standard_units
FROM data.assays
JOIN data.activities
ON data.assays.assay_id = data.activities.assay_id
WHERE data.assays.assay_organism = 'Homo sapiens' AND data.assays.assay_type = 'F'AND data.assays.cell_id IS NOT NULL 
AND data.activities.data_validity_comment is NULL AND data.activities.standard_value is not NULL AND data.activities.standard_type = 'IC50' AND data.activities.standard_units = 'nM'
);



CREATE TABLE data.human_cell_ic50 AS( 
SELECT data.temp.assay_id, data.temp.doc_id, data.temp.assay_cell_type, data.temp.cell_id, data.temp.standard_value, data.molecule_dictionary.pref_name, data.molecule_dictionary.max_phase, data.molecule_dictionary.indication_class
FROM data.temp
JOIN data.molecule_dictionary
ON data.temp.molregno = data.molecule_dictionary.molregno
WHERE data.molecule_dictionary.max_phase > 0 );