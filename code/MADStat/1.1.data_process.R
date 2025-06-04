# Load AuPapMat and PapPapMat
load('../../data/raw/MADStat/AuthorPaperInfo.RData')


# Save AuPapMat as Text file
write.csv(AuPapMat,'./derived_data/AuPapMat.txt', row.names = F)
