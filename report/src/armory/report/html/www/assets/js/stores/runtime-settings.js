import { defineStore } from 'pinia';
import { computed } from 'vue';
import { useRouter } from 'vue-router';

export const useRuntimeSettings = defineStore('runtime-settings', () => {
    const router = useRouter();
    const route = router.currentRoute;
    const updateQuery = (query) => {
        router.push({ query: { ...route.value.query, ...query } }, { replace: true });
    };

    const createSetting = (key) => computed({
        get() {
            const value = route.value.query[key];
            return value || "";
        },
        set(value) {
            updateQuery({ [key]: value });
        },
    });

    const selectedMetric = createSetting("selectedMetric");

    return { selectedMetric };
});
