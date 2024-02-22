import { RouterLink } from 'vue-router';

export default {
    components: {
        RouterLink,
    },
    props: {
        tabs: {
            type: Object,
            required: true,
        },
    },
    template: `
        <div role="tablist" class="tabs tabs-lifted">
            <router-link
                v-for="tab in tabs"
                :key="tab.dest"
                :to="{ name: tab.dest }"
                role="tab"
                class="tab"
                exact-active-class="tab-active"
            >
                {{ tab.label }}
            </router-link>
        </div>
    `,
};
