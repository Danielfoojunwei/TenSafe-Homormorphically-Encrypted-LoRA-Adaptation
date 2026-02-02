{{/*
Expand the name of the chart.
*/}}
{{- define "tensafe.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "tensafe.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "tensafe.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "tensafe.labels" -}}
helm.sh/chart: {{ include "tensafe.chart" . }}
{{ include "tensafe.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "tensafe.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tensafe.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "tensafe.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "tensafe.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Database URL
*/}}
{{- define "tensafe.databaseUrl" -}}
{{- if .Values.database.external }}
{{- printf "postgresql://%s:%s@%s:%d/%s?sslmode=%s" .Values.database.username .Values.database.password .Values.database.host (int .Values.database.port) .Values.database.name .Values.database.sslMode }}
{{- else }}
{{- printf "postgresql://tensafe:$(POSTGRES_PASSWORD)@%s-postgresql:5432/%s" (include "tensafe.fullname" .) .Values.postgresql.auth.database }}
{{- end }}
{{- end }}

{{/*
Redis URL
*/}}
{{- define "tensafe.redisUrl" -}}
{{- if .Values.redis.enabled }}
{{- printf "redis://:%s@%s-redis-master:6379/0" "$(REDIS_PASSWORD)" (include "tensafe.fullname" .) }}
{{- else }}
{{- "" }}
{{- end }}
{{- end }}
