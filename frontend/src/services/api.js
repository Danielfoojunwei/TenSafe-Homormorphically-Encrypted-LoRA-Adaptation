/**
 * TensorGuardFlow API Service Layer
 * Provides consistent API access with error handling and auth management
 */

const API_BASE = '/api/v1'

class ApiError extends Error {
    constructor(message, status, data = null) {
        super(message)
        this.status = status
        this.data = data
        this.name = 'ApiError'
    }
}

// Helper for making API requests
async function request(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`
    const config = {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        },
        ...options
    }

    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
        config.headers['Authorization'] = `Bearer ${token}`
    }

    try {
        const response = await fetch(url, config)

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}))
            throw new ApiError(
                errorData.detail || errorData.message || `HTTP ${response.status}`,
                response.status,
                errorData
            )
        }

        // Handle empty responses
        const contentType = response.headers.get('content-type')
        if (contentType && contentType.includes('application/json')) {
            return await response.json()
        }
        return null
    } catch (error) {
        if (error instanceof ApiError) throw error
        throw new ApiError(error.message, 0)
    }
}

// VLA Model Registry API
export const vlaApi = {
    listModels: (params = {}) => {
        const query = new URLSearchParams(params).toString()
        return request(`/vla/models${query ? '?' + query : ''}`)
    },
    getModel: (modelId) => request(`/vla/models/${modelId}`),
    createModel: (data) => request('/vla/models', { method: 'POST', body: JSON.stringify(data) }),
    deployModel: (modelId, fleetId, rolloutPercentage = 100) =>
        request('/vla/deploy', {
            method: 'POST',
            body: JSON.stringify({ model_id: modelId, fleet_id: fleetId, rollout_percentage: rolloutPercentage })
        }),
    submitSafetyCheck: (modelId, testScenarios = 100) =>
        request('/vla/safety/validate', {
            method: 'POST',
            body: JSON.stringify({ model_id: modelId, test_environment: 'simulation', test_scenarios: testScenarios })
        }),
    getFleetSafetyMetrics: (fleetId) => request(`/vla/safety/metrics/${fleetId}`),
    submitBenchmark: (data) => request('/vla/benchmark/submit', { method: 'POST', body: JSON.stringify(data) }),
    getBenchmarkHistory: (modelId) => request(`/vla/benchmark/${modelId}`)
}

// Identity & Certificate Management API
export const identityApi = {
    getInventory: (params = {}) => {
        const query = new URLSearchParams(params).toString()
        return request(`/identity/inventory${query ? '?' + query : ''}`)
    },
    createEndpoint: (data) => request('/identity/endpoints', { method: 'POST', body: JSON.stringify(data) }),
    requestScan: (fleetId) => request(`/identity/scan/request?fleet_id=${fleetId}`, { method: 'POST' }),
    listPolicies: () => request('/identity/policies'),
    createPolicy: (data) => request('/identity/policies', { method: 'POST', body: JSON.stringify(data) }),
    getPolicy: (policyId) => request(`/identity/policies/${policyId}`),
    runRenewals: (endpointIds, policyId) =>
        request('/identity/renewals/run', {
            method: 'POST',
            body: JSON.stringify({ endpoint_ids: endpointIds, policy_id: policyId })
        }),
    listRenewals: (params = {}) => {
        const query = new URLSearchParams(params).toString()
        return request(`/identity/renewals${query ? '?' + query : ''}`)
    },
    executeEkuMigration: () => request('/identity/migrations/eku-split', { method: 'POST' }),
    getRiskAnalysis: (certId = null) => {
        const query = certId ? `?cert_id=${certId}` : ''
        return request(`/identity/risk${query}`)
    },
    getAuditLog: (params = {}) => {
        const query = new URLSearchParams(params).toString()
        return request(`/identity/audit${query ? '?' + query : ''}`)
    },
    verifyAuditChain: () => request('/identity/audit/verify')
}

// FedMoE API
export const fedmoeApi = {
    listExperts: () => request('/fedmoe/experts'),
    createExpert: (name, baseModel) =>
        request('/fedmoe/experts', { method: 'POST', body: JSON.stringify({ name, base_model: baseModel }) }),
    getSkillsLibrary: () => request('/fedmoe/skills-library'),
    addEvidence: (expertId, evidenceType, value) =>
        request(`/fedmoe/experts/${expertId}/evidence`, {
            method: 'POST',
            body: JSON.stringify({ evidence_type: evidenceType, value })
        })
}

// Integrations API
export const integrationsApi = {
    connect: (service, config) =>
        request('/integrations/connect', { method: 'POST', body: JSON.stringify({ service, config }) }),
    getStatus: () => request('/integrations/status')
}

// TGSP Marketplace API
export const tgspApi = {
    listPackages: () => request('/tgsp/packages'),
    uploadPackage: async (file) => {
        const formData = new FormData()
        formData.append('file', file)
        const url = `${API_BASE}/tgsp/upload`
        const response = await fetch(url, { method: 'POST', body: formData })
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}))
            throw new ApiError(errorData.detail || 'Upload failed', response.status, errorData)
        }
        return response.json()
    },
    createRelease: (packageId, fleetId, channel = 'stable') =>
        request('/tgsp/releases', {
            method: 'POST',
            body: JSON.stringify({ package_id: packageId, fleet_id: fleetId, channel, is_active: true })
        }),
    getCurrentFleetPackage: (fleetId, channel = 'stable') =>
        request(`/tgsp/fleets/${fleetId}/current?channel=${channel}`)
}

// KMS API
export const kmsApi = {
    getKeys: () => request('/kms/keys'),
    getRotationSchedule: () => request('/kms/rotation-schedule'),
    getAttestationPolicies: () => request('/kms/attestation-policies'),
    rotateKey: (kid, reason = 'manual_rotation') =>
        request('/kms/rotate', { method: 'POST', body: JSON.stringify({ kid, reason }) })
}

// Pipeline Config API
export const pipelineApi = {
    getConfig: () => request('/pipeline/config'),
    updateConfig: (key, value) =>
        request('/pipeline/config', { method: 'PUT', body: JSON.stringify({ key, value: String(value) }) }),
    resetConfig: () => request('/pipeline/config/reset', { method: 'POST' })
}

// Bayesian Policy API
export const policyApi = {
    getBayesianConfig: () => request('/policy/bayesian/config'),
    updateRules: (rules) => request('/policy/bayesian/rules', { method: 'POST', body: JSON.stringify(rules) }),
    triggerEvaluation: (runId) =>
        request(`/policy/bayesian/evaluate?run_id=${runId}`, { method: 'POST' })
}

// Forensics API
export const forensicsApi = {
    getIncidents: () => request('/forensics/incidents'),
    analyzeIncident: (incidentId, timeWindowHours = 24) =>
        request('/forensics/analyze', {
            method: 'POST',
            body: JSON.stringify({ incident_id: incidentId, time_window_hours: timeWindowHours })
        }),
    verifyCompliance: () => request('/forensics/verify-compliance', { method: 'POST' })
}

// Fleet API
export const fleetApi = {
    listFleets: () => request('/fleets/extended'),
    createFleet: (name) => request(`/fleets?name=${encodeURIComponent(name)}`, { method: 'POST' })
}

// PEFT API
export const peftApi = {
    getProfiles: () => request('/peft/profiles'),
    listRuns: () => request('/peft/runs'),
    createRun: (wizardState) => request('/peft/runs', { method: 'POST', body: JSON.stringify(wizardState) }),
    getRun: (runId) => request(`/peft/runs/${runId}`),
    promoteRun: (runId, channel) =>
        request(`/peft/runs/${runId}/promote`, { method: 'POST', body: JSON.stringify({ channel }) })
}

// Status & Health API
export const statusApi = {
    getStatus: () => request('/status'),
    getHealth: () => request('/health')
}

// TG-Tinker Training API
export const tinkerApi = {
    // Training Clients
    createTrainingClient: (config) =>
        request('/v1/training_clients', { method: 'POST', body: JSON.stringify(config) }),
    getTrainingClient: (clientId) => request(`/v1/training_clients/${clientId}`),
    listTrainingClients: (params = {}) => {
        const query = new URLSearchParams(params).toString()
        return request(`/v1/training_clients${query ? '?' + query : ''}`)
    },

    // Async Operations
    getFuture: (futureId) => request(`/v1/futures/${futureId}`),
    cancelFuture: (futureId) => request(`/v1/futures/${futureId}/cancel`, { method: 'POST' }),

    // Training Operations
    forwardBackward: (clientId, batch) =>
        request(`/v1/training_clients/${clientId}/forward_backward`, {
            method: 'POST',
            body: JSON.stringify(batch)
        }),
    optimStep: (clientId, options = {}) =>
        request(`/v1/training_clients/${clientId}/optim_step`, {
            method: 'POST',
            body: JSON.stringify(options)
        }),
    sample: (clientId, prompts, options = {}) =>
        request(`/v1/training_clients/${clientId}/sample`, {
            method: 'POST',
            body: JSON.stringify({ prompts, ...options })
        }),

    // State Management
    saveState: (clientId, options = {}) =>
        request(`/v1/training_clients/${clientId}/save_state`, {
            method: 'POST',
            body: JSON.stringify(options)
        }),
    loadState: (clientId, artifactId) =>
        request(`/v1/training_clients/${clientId}/load_state`, {
            method: 'POST',
            body: JSON.stringify({ artifact_id: artifactId })
        }),

    // Artifacts
    getArtifact: (artifactId) => request(`/v1/artifacts/${artifactId}`),
    listArtifacts: (clientId = null, params = {}) => {
        if (clientId) params.training_client_id = clientId
        const query = new URLSearchParams(params).toString()
        return request(`/v1/artifacts${query ? '?' + query : ''}`)
    },

    // Audit Logs
    getAuditLogs: (params = {}) => {
        const query = new URLSearchParams(params).toString()
        return request(`/v1/audit_logs${query ? '?' + query : ''}`)
    },
    verifyAuditChain: (tenantId = null) => {
        const query = tenantId ? `?tenant_id=${tenantId}` : ''
        return request(`/v1/audit_logs/verify${query}`)
    },

    // Privacy Metrics
    getPrivacySpent: (clientId) => request(`/v1/training_clients/${clientId}/privacy_spent`),

    // TGSP Export
    exportToTGSP: (clientId, artifactId, options = {}) =>
        request(`/v1/training_clients/${clientId}/export_tgsp`, {
            method: 'POST',
            body: JSON.stringify({ artifact_id: artifactId, ...options })
        })
}

export { ApiError, request }
