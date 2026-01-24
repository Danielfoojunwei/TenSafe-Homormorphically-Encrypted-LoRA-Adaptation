import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useContinuousStore = defineStore('continuous', () => {
    // State
    const routes = ref([])
    const selectedRoute = ref(null)
    const routeDetails = ref({})
    const timeline = ref({})
    const loading = ref(false)
    const showWizard = ref(false)
    const error = ref(null)
    const metricsSummary = ref([])
    const dashboardBundle = ref(null)

    // Stats
    const stats = ref({
        adaptersThisWeek: 0,
        rollbackReady: 0,
    })

    // Computed
    const scheduledUpdates = computed(() => {
        return routes.value
            .filter(r => r.next_scheduled_at)
            .map(r => ({
                route_key: r.route_key,
                next_at: new Date(r.next_scheduled_at).toLocaleString()
            }))
    })

    // Actions
    const fetchRoutes = async (tenantId = 'default') => {
        loading.value = true
        error.value = null
        try {
            const res = await fetch('/api/v1/tgflow/routes')
            if (res.ok) {
                routes.value = await res.json()
                // Calculate stats
                stats.value.rollbackReady = routes.value.filter(r => r.fallback_adapter_id).length
            } else {
                throw new Error('Failed to fetch routes')
            }

            // Also fetch summary
            const summaryRes = await fetch(`/api/v1/metrics/routes/summary?tenant_id=${tenantId}`)
            if (summaryRes.ok) {
                metricsSummary.value = await summaryRes.json()
            }
        } catch (e) {
            console.warn('Failed to fetch routes', e)
            error.value = e.message
            // Demo data for development
            routes.value = [
                {
                    route_key: 'customer-support',
                    base_model_ref: 'microsoft/phi-2',
                    enabled: true,
                    stage: 'stable',
                    adapter_count: 3,
                    active_adapter_id: 'adapter-001',
                    last_loop_at: new Date().toISOString(),
                    privacy_mode: 'off',
                }
            ]
        } finally {
            loading.value = false
        }
    }

    const fetchDashboardBundle = async (routeKey, tenantId = 'default') => {
        try {
            const res = await fetch(`/api/v1/metrics/routes/${routeKey}/dashboard_bundle?tenant_id=${tenantId}`)
            if (res.ok) {
                dashboardBundle.value = await res.json()
                return dashboardBundle.value
            }
        } catch (e) {
            console.warn('Failed to fetch dashboard bundle', e)
        }
        return null
    }

    const exportSnapshot = async (routeKey, tenantId = 'default') => {
        try {
            const bundle = await fetchDashboardBundle(routeKey, tenantId)
            if (bundle) {
                const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' })
                const url = URL.createObjectURL(blob)
                const link = document.createElement('a')
                link.href = url
                link.download = `tg_analytics_${routeKey}_${new Date().toISOString().split('T')[0]}.json`
                document.body.appendChild(link)
                link.click()
                document.body.removeChild(link)
                URL.revokeObjectURL(url)
            }
        } catch (e) {
            console.warn('Failed to export snapshot', e)
        }
    }

    const fetchRouteDetails = async (routeKey) => {
        try {
            const res = await fetch(`/api/v1/tgflow/routes/${routeKey}`)
            if (res.ok) {
                routeDetails.value[routeKey] = await res.json()
            }
        } catch (e) {
            console.warn('Failed to fetch route details', e)
        }
    }

    const fetchTimeline = async (routeKey) => {
        try {
            const res = await fetch(`/api/v1/tgflow/routes/${routeKey}/timeline`)
            if (res.ok) {
                const data = await res.json()
                timeline.value[routeKey] = data.timeline || []
            }
        } catch (e) {
            console.warn('Failed to fetch timeline', e)
            // Demo timeline
            timeline.value[routeKey] = [
                {
                    loop_id: 'loop-001',
                    trigger: 'manual',
                    started_at: new Date().toISOString(),
                    verdict: 'success',
                    summary: 'Adapter adapter-002 produced and registered',
                    adapter_produced: 'adapter-002',
                    events: [
                        { stage: 'ingest', headline: 'Data ingested successfully', explanation: 'Feed snapshot captured (1000 records).', verdict: 'success', duration_ms: 120 },
                        { stage: 'novelty_check', headline: 'Update proposed', explanation: 'Data novelty (0.65) exceeds threshold (0.30).', verdict: 'success', duration_ms: 50 },
                        { stage: 'train', headline: 'Training complete', explanation: 'Adapter trained for 3 epochs. Final loss: 0.15', verdict: 'success', duration_ms: 8500 },
                        { stage: 'eval', headline: 'Evaluation passed', explanation: 'Quality: 94%. Forgetting: 3%. Regression: 1%.', verdict: 'success', duration_ms: 200 },
                        { stage: 'package', headline: 'TGSP package created', explanation: 'Package hash: a1b2c3d4e5f6...', verdict: 'success', duration_ms: 80 },
                        { stage: 'register', headline: 'Adapter registered', explanation: 'Adapter adapter-002 registered to route.', verdict: 'success', duration_ms: 30 },
                    ]
                }
            ]
        }
    }

    const createRoute = async (formData) => {
        try {
            // Create route
            const routeRes = await fetch('/api/v1/tgflow/routes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    route_key: formData.route_key,
                    base_model_ref: formData.base_model_ref,
                    description: formData.description,
                })
            })

            if (!routeRes.ok) {
                throw new Error('Failed to create route')
            }

            // Connect feed
            await fetch(`/api/v1/tgflow/routes/${formData.route_key}/feed`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    feed_type: formData.feed_type,
                    feed_uri: formData.feed_uri,
                    privacy_mode: formData.privacy_mode,
                })
            })

            // Set policy
            await fetch(`/api/v1/tgflow/routes/${formData.route_key}/policy`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    novelty_threshold: formData.novelty_threshold,
                    forgetting_budget: formData.forgetting_budget,
                    regression_budget: formData.regression_budget,
                    update_cadence: formData.update_cadence,
                    auto_promote_to_canary: formData.auto_promote_to_canary,
                })
            })

            return true
        } catch (e) {
            console.error('Failed to create route', e)
            error.value = e.message
            return false
        }
    }

    const runOnce = async (routeKey) => {
        try {
            const res = await fetch(`/api/v1/tgflow/routes/${routeKey}/run_once`, {
                method: 'POST'
            })
            if (res.ok) {
                const data = await res.json()
                // Refresh timeline
                await fetchTimeline(routeKey)
                return data
            }
        } catch (e) {
            console.warn('Failed to run loop', e)
        }
        return null
    }

    const promote = async (routeKey, adapterId, target) => {
        try {
            const res = await fetch(`/api/v1/tgflow/routes/${routeKey}/promote?adapter_id=${adapterId}&target=${target}`, {
                method: 'POST'
            })
            if (res.ok) {
                await fetchRoutes()
                return await res.json()
            }
        } catch (e) {
            console.warn('Promote failed', e)
        }
        return { ok: false }
    }

    const rollback = async (routeKey) => {
        try {
            const res = await fetch(`/api/v1/tgflow/routes/${routeKey}/rollback`, {
                method: 'POST'
            })
            if (res.ok) {
                await fetchRoutes()
                return await res.json()
            }
        } catch (e) {
            console.warn('Rollback failed', e)
        }
        return { ok: false }
    }

    const fetchDiff = async (routeKey, fromAdapter, toAdapter) => {
        try {
            let url = `/api/v1/tgflow/routes/${routeKey}/diff`
            if (fromAdapter) url += `?from_adapter=${fromAdapter}`
            if (toAdapter) url += `&to_adapter=${toAdapter}`

            const res = await fetch(url)
            if (res.ok) {
                return await res.json()
            }
        } catch (e) {
            console.warn('Diff failed', e)
        }
        return { diff_available: false }
    }

    const exportRoute = async (routeKey, backend = 'k8s') => {
        try {
            const res = await fetch(`/api/v1/tgflow/routes/${routeKey}/export?backend=${backend}`, {
                method: 'POST'
            })
            if (res.ok) {
                return await res.json()
            }
        } catch (e) {
            console.warn('Export failed', e)
        }
        return null
    }

    return {
        // State
        routes,
        selectedRoute,
        routeDetails,
        timeline,
        loading,
        showWizard,
        error,
        stats,
        metricsSummary,
        dashboardBundle,

        // Computed
        scheduledUpdates,

        // Actions
        fetchRoutes,
        fetchRouteDetails,
        fetchDashboardBundle,
        exportSnapshot,
        fetchTimeline,
        createRoute,
        runOnce,
        promoteAdapter: promote,
        rollbackRoute: rollback,
        fetchDiff,
        exportRoute,
    }
})
